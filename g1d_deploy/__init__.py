"""g1d_deploy —— 宇树 G1-D 实体机器人「抓取」真机部署包。

本包把工作区里已有的三块能力串成一条可上真机的链路：

    VLA 策略服务器 (GPU, /act)
          ▲  HTTP (观测)
          │
    ┌─────┴──────────────────────────────────────────┐
    │  G1RealRobot  (实现 arena.RobotInterface)       │
    │   ├─ 观测: camera.py + 低层 lowstate 关节状态     │
    │   ├─ 动作: action_mapper.py (关节空间 / EE+IK)    │
    │   └─ 执行: g1_lowlevel.py (500Hz DDS 关节控制)    │
    └────────────────────────────────────────────────┘

设计原则
------------------------------------------------------------------
* **只依赖 unitree_sdk2_python 与 numpy**，不依赖 Isaac Lab（真机不需要仿真器）；
* **复用 ARENA 框架的 `RobotInterface` / `PolicyClient` / `EmbodimentAdapter`**，
  因此与仿真侧 `arena_g1_locomanip_pnp.py` 共享同一条闭环逻辑；
* **安全优先**：默认只驱动双臂 + 夹爪，下肢保持当前姿态；内置关节限位、增益上限、
  平滑插值、看门狗与急停；
* **可插拔**：IK、相机、夹爪槽位均为显式扩展点，且默认给出可离线自测的最小实现。

⚠️ 真机前必读 `README.md`：本包很多参数（关节方向、夹爪槽位、相机、归一化统计、
网卡/域）必须按实际硬件标定，代码中用 ``FIXME(标定)`` 显式标出。
"""

from .joint_map import (
    G1JointIndex,
    G1_NUM_MOTOR,
    HG_NUM_MOTOR,
    JOINT_ACTION_DIM,
    EE_ACTION_DIM,
    DEFAULT_KP,
    DEFAULT_KD,
    ARM_JOINT_LIMITS,
)

__all__ = [
    "G1JointIndex",
    "G1_NUM_MOTOR",
    "HG_NUM_MOTOR",
    "JOINT_ACTION_DIM",
    "EE_ACTION_DIM",
    "DEFAULT_KP",
    "DEFAULT_KD",
    "ARM_JOINT_LIMITS",
]
