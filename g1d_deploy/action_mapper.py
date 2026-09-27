"""VLA 动作 → G1 关节目标 的映射层。

策略输出的动作有两种口径（见 ``joint_map.py`` 与 DOCUMENTATION §12.2）：

1. **关节空间（16 维）**：``[左臂7, 右臂7, 左夹爪, 右夹爪]`` —— 直接映射到电机槽位，
   无需逆运动学，是**真机首次上电最稳妥**的路径；
2. **EE 末端位姿（23 维）**：``[导航…, 左EEF位姿, 右EEF位姿, …]`` —— 需要**逆运动学
   (IK)** 把末端位姿解算成 7 个臂关节角。

本模块把"两种口径"统一成一个接口 :class:`ActionMapper.map`，上层闭环只认这一个
接口，切换口径只需换映射器实例。

IK 的诚实说明
------------------------------------------------------------------
仿真侧（``arena_g1_locomanip_pnp.py``）用的是 Isaac Lab Arena 内置的 **PINK IK**；
真机侧 unitree SDK **不提供** IK 服务。因此 EE 路径需要自带 IK：

* 可在机器人上位机复用一个 URDF 数值 IK 求解器（如 ``ikpy``），
* 或把仿真里已标定好的 PINK IK 包装成 :class:`G1IK` 实现。

本文件只定义 IK 接口与一个**恒等调试桩**（``DebugIK``，用于冒烟测试跑通链路），
真实 IK 由使用者在 ``EEActionMapper(ik=...)`` 注入。关节空间路径不依赖 IK。
"""

from __future__ import annotations

import abc
from typing import Dict, Optional

import numpy as np

from .joint_map import (
    EE_ACTION_DIM,
    split_joint_action,
)


# ============================================================================
# IK 接口（EE 路径用）
# ============================================================================


class G1IK(abc.ABC):
    """G1 单臂逆运动学求解接口。

    实现者需要把"腕部末端在基座坐标系下的位姿"解算成 7 个臂关节角。
    返回 ``None`` 表示该位姿不可达（上层应丢弃该动作并保持当前姿态）。
    """

    @abc.abstractmethod
    def solve_left(self, position: np.ndarray, quat_wxyz: np.ndarray) -> Optional[np.ndarray]:
        """解算左臂 7 关节角（弧度）。

        Args:
            position: ``(3,)`` 末端位置。
            quat_wxyz: ``(4,)`` 末端姿态四元数 ``[w, x, y, z]``。

        Returns:
            ``(7,)`` 关节角，或 ``None``（不可达）。
        """

    @abc.abstractmethod
    def solve_right(self, position: np.ndarray, quat_wxyz: np.ndarray) -> Optional[np.ndarray]:
        """解算右臂 7 关节角（弧度），参数/返回值同 :meth:`solve_left`。"""


class DebugIK(G1IK):
    """恒等调试桩：返回"保持当前姿态"的关节角（用于冒烟测试，不做真 IK）。

    它把传入的末端位姿**忽略**，直接返回 ``current_q``。用途是验证
    "EE 动作 → 映射 → 低层控制" 整条链路能跑通，而不需要真实 IK。
    """

    def __init__(self, left_q: np.ndarray, right_q: np.ndarray) -> None:
        self.left_q = np.asarray(left_q, dtype=np.float32).reshape(7)
        self.right_q = np.asarray(right_q, dtype=np.float32).reshape(7)

    def solve_left(self, position, quat_wxyz):
        return self.left_q.copy()

    def solve_right(self, position, quat_wxyz):
        return self.right_q.copy()


# ============================================================================
# 映射器
# ============================================================================


class ActionMapper(abc.ABC):
    """把单步 VLA 动作映射为 ``{left_arm, right_arm, left_gripper, right_gripper}``。"""

    @abc.abstractmethod
    def map(self, action: np.ndarray, current_arm_state: np.ndarray) -> Dict[str, np.ndarray]:
        """映射单个动作。

        Args:
            action: 模型输出的单步动作（未反归一化前的物理量，或已反归一化后均可，
                由具体实现约定）。
            current_arm_state: 当前双臂实测关节角 ``[左臂7, 右臂7]``（弧度），
                供 IK / 相对动作使用。

        Returns:
            ``{"left_arm": (7,), "right_arm": (7,), "left_gripper": 标量, "right_gripper": 标量}``。
        """


class JointSpaceMapper(ActionMapper):
    """16 维关节空间动作映射（默认，最稳妥）。

    直接把动作向量拆成臂关节目标与夹爪命令；夹爪命令用
    :func:`~g1d_deploy.joint_map.gripper_cmd_to_position` 线性映射到电机位置。
    """

    def map(self, action: np.ndarray, current_arm_state: np.ndarray) -> Dict[str, np.ndarray]:
        parts = split_joint_action(np.asarray(action, dtype=np.float32).reshape(-1))
        return {
            "left_arm": parts["left_arm"].astype(np.float32),
            "right_arm": parts["right_arm"].astype(np.float32),
            "left_gripper": np.asarray(parts["left_gripper"], dtype=np.float32),
            "right_gripper": np.asarray(parts["right_gripper"], dtype=np.float32),
        }


class EEActionMapper(ActionMapper):
    """23 维 EE 末端位姿动作映射（需注入 :class:`G1IK`）。

    动作布局（见 DOCUMENTATION §12.2）：
    ``2:5`` 左EEF位置、``5:9`` 左EEF四元数(wxyz)、``9:12`` 右EEF位置、
    ``12:16`` 右EEF四元数(wxyz)。其余维度（导航/底座/躯干）在"固定站立抓取"
    场景下被忽略。

    夹爪不在 23 维动作里，因此由 :attr:`gripper_policy` 提供 —— 默认
    :meth:`_default_gripper_policy` 恒"张开"（0），安全起见不自动闭合。
    """

    def __init__(self, ik: G1IK, gripper_policy=None) -> None:
        self.ik = ik
        self.gripper_policy = gripper_policy or self._default_gripper_policy

    @staticmethod
    def _default_gripper_policy(_action, _current_state) -> tuple:
        """默认夹爪策略：恒张开。抓取时机需由上层或自定义策略决定。"""
        return 0.0, 0.0

    def map(self, action: np.ndarray, current_arm_state: np.ndarray) -> Dict[str, np.ndarray]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < EE_ACTION_DIM:
            raise ValueError(f"EE action dim {action.shape[0]} < {EE_ACTION_DIM}")

        left_pos = action[2:5]
        left_quat = action[5:9]
        right_pos = action[9:12]
        right_quat = action[12:16]

        left_q = self.ik.solve_left(left_pos, left_quat)
        right_q = self.ik.solve_right(right_pos, right_quat)
        if left_q is None or right_q is None:
            # 不可达：退回保持当前姿态（安全）
            left_q = current_arm_state[0:7]
            right_q = current_arm_state[7:14]

        left_g, right_g = self.gripper_policy(action, current_arm_state)
        return {
            "left_arm": np.asarray(left_q, dtype=np.float32).reshape(7),
            "right_arm": np.asarray(right_q, dtype=np.float32).reshape(7),
            "left_gripper": np.asarray(left_g, dtype=np.float32),
            "right_gripper": np.asarray(right_g, dtype=np.float32),
        }


def build_action_mapper(mode: str, ik: Optional[G1IK] = None, **kwargs) -> ActionMapper:
    """按 ``mode`` 构造映射器。

    Args:
        mode: ``"joint"``（16 维关节空间，默认）或 ``"ee"``（23 维末端位姿）。
        ik: EE 模式所需的 IK 实例；``None`` 时用 :class:`DebugIK` 桩。

    Returns:
        :class:`ActionMapper` 实例。
    """
    if mode == "joint":
        return JointSpaceMapper()
    if mode == "ee":
        ik = ik or DebugIK(np.zeros(7, dtype=np.float32), np.zeros(7, dtype=np.float32))
        return EEActionMapper(ik, **kwargs)
    raise ValueError(f"未知动作模式: {mode}（可选 'joint' / 'ee'）")
