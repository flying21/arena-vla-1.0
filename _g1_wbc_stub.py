#!/usr/bin/env python3
# 麻雀虽小智能科技（武汉）有限公司
"""G1 WBC 下肢策略 Stub —— 替代无法访问的 NVIDIA 专有 ONNX 模型。

背景
------------------------------------------------------------------
Isaac Lab Arena 的 G1 移动操作实施例依赖 ``G1HomiePolicyV2`` 作为**下肢
运动策略**。该策略需要 ``stand.onnx`` / ``walk.onnx`` 两个神经网络权重，
存放在 NVIDIA 内部 Nucleus 服务器上::

    omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/
        g1_locomanip_assets/wbc_policy/

该地址在当前环境**不可达**，因此用本文件的固定站立姿态 Stub 顶替，使得：

* 环境**能够被构建**（资产加载不再因缺 ONNX 而中断）；
* Scene / Embodiment / Task 三模块的组合与 PnP 任务逻辑**可被验证**；
* 上肢仍然由真实的 PINK IK 控制，因此"抓取-放置"的上半身链路是真的。

必须明确的边界（诚实声明）
------------------------------------------------------------------
本 Stub **不等价于**真实 WBC 策略。它只输出恒定站立姿态，因此：

* 机器人**不会行走**，导航子目标（``navigate_cmd``）不会被真正执行；
* 因此 Mock 模式下 PnP 任务的 ``success=false`` 是**符合预期**的，
  它说明"仿真与接口链路能跑通"，而不说明"任务已完成"。

这一点在技术文档中被反复强调，避免把"接口可验证"误读成"任务已实现"。
"""

import logging

import numpy as np

logger = logging.getLogger("arena")

#: 固定站立姿态的下肢关节角（弧度）。
#:
#: ⚠️ 维度为 **15**，而 ``DOCUMENTATION.md`` 里描述下肢为 "6 DOF × 2 = 12 维"。
#: 二者不一致，说明这里的关节排列是按 Arena SDK 的 ``body_action`` 契约
#: （含腰部/躯干关节）写的。若接入真实 WBC，应以 SDK 的实际契约为准重新核对。
DEFAULT_LOWER_BODY_ANGLES = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
     -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
      0.0, 0.0, 0.0],
    dtype=np.float32,
)


class StubHomiePolicyV2:
    """返回固定站立姿态的下肢策略，用于替代 ONNX 推理。

    接口刻意与 ``G1HomiePolicyV2`` 保持一致（鸭子类型替换），
    因此可以被直接赋值到工厂模块的类名上（见 :func:`apply_patch`）：

    ==========================  ==============================================
    方法                         本 Stub 的行为
    ==========================  ==============================================
    ``__init__``                只记录 ``num_envs``，不加载任何模型
    ``set_goal``                记录导航指令 ``navigate_cmd``（但不执行）
    ``set_observation``         记录观测（但不使用）
    ``reset``                   把导航指令清零
    ``get_action``              恒定返回平铺的站立姿态
    ``close``                   空实现（无资源可释放）
    ==========================  ==============================================
    """

    def __init__(self, robot_model=None, config_path=None, model_path=None, num_envs=1):
        """初始化。

        Args:
            robot_model: 机器人模型对象（Stub 不使用，仅为接口兼容）。
            config_path: 策略配置路径（Stub 不使用）。
            model_path: ONNX 权重路径（Stub 不使用 —— 正因为它不可达才有本类）。
            num_envs: 并行环境数，决定 ``get_action`` 返回的批大小。
        """
        self.robot_model = robot_model
        self.num_envs = int(num_envs)
        self.use_policy_action = True  # 告诉上层"我提供动作"
        self.cmd = np.zeros(3, dtype=np.float32)  # 导航速度/朝向指令
        self.height_cmd = 0.74  # 底座高度指令（米），G1 站立高度
        self.observation = None

    def set_goal(self, goal):
        """接收上层下发的目标；只提取 ``navigate_cmd`` 并记录。"""
        if isinstance(goal, dict) and "navigate_cmd" in goal:
            self.cmd = goal["navigate_cmd"]

    def set_observation(self, observation):
        """接收观测并缓存（真实策略会用它做推理，Stub 不需要）。"""
        self.observation = observation

    def reset(self, env_ids=None):
        """复位时清空导航指令（站立不动）。"""
        self.cmd = np.zeros(3, dtype=np.float32)

    def get_action(self, *args, **kwargs):
        """返回恒定站立动作。

        Returns:
            ``{"body_action": ndarray(num_envs, 15)}`` —— 把同一个站立姿态
            在批维度平铺，形状与真实策略的输出契约一致。
        """
        return {"body_action": np.tile(DEFAULT_LOWER_BODY_ANGLES, (self.num_envs, 1))}

    def close(self):
        """空实现：Stub 没有需要释放的资源。"""
        pass


def apply_patch() -> bool:
    """把工厂模块里的 ``G1HomiePolicyV2`` 替换为 :class:`StubHomiePolicyV2`。

    实现方式是**直接改写模块属性**：Arena SDK 内部通过
    ``wbc_policy_factory.G1HomiePolicyV2`` 查表构造策略，
    因此替换这个类属性即可让所有后续实例化都拿到 Stub，无需改动 SDK 源码。

    Returns:
        True 表示替换成功；False 表示 Arena SDK 未安装或导入失败。

    注意: 失败时**只记录警告、不抛异常**。这样脚本能继续往下跑，但故障会在
        稍后以"其它不相关的报错"形式出现。排障时若日志里有
        ``⚠ WBC Stub patch 失败``，应优先解决它。
    """
    try:
        import isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.wbc_policy_factory as factory
        factory.G1HomiePolicyV2 = StubHomiePolicyV2
        logger.info("✓ WBC 下肢策略已替换为本地 Stub (固定站立姿态)")
        return True
    except Exception as e:
        logger.warning("⚠ WBC Stub patch 失败: %s", e)
        return False
