"""G1-D 关节索引、动作空间与电机增益/限位的统一口径。

本模块是全包**最底层、最稳定**的常量表，其它模块（低层控制器、动作映射、真机适配）
都从这里取数，避免同一份关节编号在多个文件里漂移。

数据来源与可信度说明
------------------------------------------------------------------
* :class:`G1JointIndex` 的电机槽位编号来自 SDK 官方例程
  ``unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py``，属于固件
  契约，可信度最高；
* ``DEFAULT_KP`` / ``DEFAULT_KD`` 同样来自该例程（仅为"能安全带动关节"的演示值，
  **不是最优值**，真机需按负载重新整定）；
* ``ARM_JOINT_LIMITS`` 是**保守安全限位**（弧度），用于动作裁剪兜底；真实机械限位
  请以 URDF（``g1_29dof_with_hand_rev_1_0.urdf``）为准并覆盖此表；
* 夹爪/手部槽位（``GRIPPER_SLOTS``）当前 SDK 例程**未给出**，此处为占位默认，
  必须按实际 G1-D 手部接线标定 —— 见 README 的标定清单。

动作空间（两种，与本工作区仿真/训练口径一致）
------------------------------------------------------------------
============  ======  ====================================================
名称          维度     含义
============  ======  ====================================================
关节空间       16      [左臂7, 右臂7, 左夹爪1, 右夹爪1]
EE 末端位姿    23      见 :data:`EE_ACTION_DIM` 与 README §动作空间
============  ======  ====================================================

16 维关节动作的臂关节顺序（默认，可用 ``ARM_JOINT_ORDER`` 覆盖）：
肩 pitch/roll/yaw → 肘 → 腕 roll/pitch/yaw，与 :class:`G1JointIndex` 一一对应。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# ============================================================================
# 电机槽位总数
# ============================================================================

#: G1（29 自由度）全身电机槽位总数（腿 12 + 腰 3 + 臂 14）。
G1_NUM_MOTOR = 29

#: unitree_hg 的 LowCmd/LowState 实际电机槽位数（29 个 + 6 个预留/手部槽位）。
#: crc.py 的打包格式串里正是 ``... * 35``；手部/夹爪通常落在 29..34。
HG_NUM_MOTOR = 35

# ============================================================================
# 关节索引（固件契约，勿改）
# ============================================================================


class G1JointIndex:
    """G1 关节 → 低层消息 ``motor_cmd``/``motor_state`` 数组下标（0..28）。

    下标顺序按固件侧电机关节编号固定：左腿 0~5、右腿 6~11、腰 12~14、
    左臂 15~21、右臂 22~28。多个常量指向同一槽位（如 ``LeftAnklePitch`` 与
    ``LeftAnkleB`` 都是 4）是因为 PR/AB 两种控制模式下对同一物理关节的叫法不同。
    """

    # ---- 左腿（0~5）----
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # ---- 右腿（6~11）----
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    # ---- 腰（12~14）----
    WaistYaw = 12
    WaistRoll = 13   # 23dof/29dof 锁腰时无效
    WaistA = 13
    WaistPitch = 14  # 23dof/29dof 锁腰时无效
    WaistB = 14

    # ---- 左臂（15~21）----
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # 23dof 无效
    LeftWristYaw = 21     # 23dof 无效

    # ---- 右臂（22~28）----
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # 23dof 无效
    RightWristYaw = 28    # 23dof 无效


#: 左臂 7 关节的电机槽位（顺序：肩 pitch/roll/yaw、肘、腕 roll/pitch/yaw）。
LEFT_ARM_SLOTS: List[int] = [
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
]

#: 右臂 7 关节的电机槽位（顺序同左臂）。
RIGHT_ARM_SLOTS: List[int] = [
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
]

#: 抓取时**默认不主动驱动**的下半身槽位（腿 + 腰）。它们会被"保持当前姿态"。
LOWER_BODY_SLOTS: List[int] = list(range(0, 15))

# ============================================================================
# 动作空间维度
# ============================================================================

#: 关节空间动作维度：左臂 7 + 右臂 7 + 左夹爪 1 + 右夹爪 1。
JOINT_ACTION_DIM = 16

#: EE 末端位姿动作维度（与 DOCUMENTATION §12.2 / constants.G1_EE_6D 一致）：
#: 0-1 导航速度, 2-4 左EEF位置, 5-8 左EEF四元数(wxyz), 9-11 右EEF位置,
#: 12-15 右EEF四元数(wxyz), 16-18 导航子目标, 19 底座高度, 20-22 躯干 RPY。
EE_ACTION_DIM = 23

#: 16 维关节动作的默认排列（真机标定时可覆盖）。
#: 索引 0..6 左臂、7..13 右臂、14 左夹爪、15 右夹爪。
ARM_JOINT_ORDER: List[str] = [
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    "left_gripper", "right_gripper",
]

#: 关节空间动作中"左臂 7 维"所在的下标范围（闭开区间 [0,7)）。
LEFT_ARM_ACTION_SLICE: Tuple[int, int] = (0, 7)
#: 右臂 7 维。
RIGHT_ARM_ACTION_SLICE: Tuple[int, int] = (7, 14)
#: 左夹爪下标。
LEFT_GRIPPER_ACTION_IDX: int = 14
#: 右夹爪下标。
RIGHT_GRIPPER_ACTION_IDX: int = 15

# ============================================================================
# 增益与限位
# ============================================================================

#: 位置增益 Kp（与例程一致，仅演示值，真机需整定）。
#: 前 12 腿、中 3 腰、后 14 臂。臂关节统一 40，夹爪槽位另配。
DEFAULT_KP: List[float] = [
    60, 60, 60, 100, 40, 40,
    60, 60, 60, 100, 40, 40,
    60, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
    40, 40, 40, 40, 40, 40, 40,
]

#: 速度增益 Kd（与例程一致，仅演示值）。
DEFAULT_KD: List[float] = [
    1, 1, 1, 2, 1, 1,
    1, 1, 1, 2, 1, 1,
    1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
]

#: 双臂 14 关节的**保守软限位**（弧度），用于动作裁剪兜底。
#: 顺序与 ``LEFT_ARM_SLOTS + RIGHT_ARM_SLOTS`` 一致；真实限位以 URDF 为准。
#: FIXME(标定): 以下为占位值，请按 g1_29dof_with_hand_rev_1_0.urdf 覆盖。
ARM_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "shoulder_pitch": (-2.0, 2.0),
    "shoulder_roll": (-1.2, 1.2),
    "shoulder_yaw": (-2.0, 2.0),
    "elbow": (-2.6, 2.6),
    "wrist_roll": (-2.0, 2.0),
    "wrist_pitch": (-1.6, 1.6),
    "wrist_yaw": (-2.0, 2.0),
}

#: 夹爪动作值 → 电机位置目标的最小/最大映射（约定 0=张开、1=闭合，可标定）。
#: FIXME(标定): 槽位与方向必须按实际手部标定，见 README。
GRIPPER_SLOTS: Dict[str, int] = {"left": 29, "right": 30}
GRIPPER_POS_RANGE: Tuple[float, float] = (0.0, 1.0)


# ============================================================================
# 映射辅助函数（纯函数，可离线单测）
# ============================================================================


def arm_slots_flat() -> List[int]:
    """返回 ``左臂7 + 右臂7`` 的电机槽位拼接（共 14 维）。"""
    return LEFT_ARM_SLOTS + RIGHT_ARM_SLOTS


def split_joint_action(action: np.ndarray) -> Dict[str, np.ndarray]:
    """把 16 维关节动作拆成 ``{left_arm, right_arm, left_gripper, right_gripper}``。

    Args:
        action: 形状 ``(16,)`` 的关节动作向量。

    Returns:
        各分量的 numpy 数组；``left_arm``/``right_arm`` 各 7 维，夹爪各 1 维标量。
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] < JOINT_ACTION_DIM:
        raise ValueError(f"joint action dim {action.shape[0]} < {JOINT_ACTION_DIM}")
    return {
        "left_arm": action[LEFT_ARM_ACTION_SLICE[0]:LEFT_ARM_ACTION_SLICE[1]],
        "right_arm": action[RIGHT_ARM_ACTION_SLICE[0]:RIGHT_ARM_ACTION_SLICE[1]],
        "left_gripper": float(action[LEFT_GRIPPER_ACTION_IDX]),
        "right_gripper": float(action[RIGHT_GRIPPER_ACTION_IDX]),
    }


def clip_arm_action(arm: np.ndarray) -> np.ndarray:
    """按 :data:`ARM_JOINT_LIMITS` 裁剪单个 7 维臂动作（按关节名限位）。"""
    arm = np.asarray(arm, dtype=np.float32).reshape(-1)
    names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
             "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
    out = arm.copy()
    for i, name in enumerate(names):
        low, high = ARM_JOINT_LIMITS[name]
        out[i] = float(np.clip(arm[i], low, high))
    return out


def gripper_cmd_to_position(cmd: float) -> float:
    """把夹爪动作值（约定 0=张开、1=闭合）映射到电机位置目标。

    夹爪动作可能来自模型 ``[-1, 1]`` 反归一化后的任意值，先裁剪到 [0,1] 再线性
    映射到 :data:`GRIPPER_POS_RANGE`。
    """
    cmd = float(np.clip(cmd, 0.0, 1.0))
    low, high = GRIPPER_POS_RANGE
    return low + cmd * (high - low)
