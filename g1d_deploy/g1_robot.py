"""G1-D 真机适配：把低层控制器 + 相机 + 动作映射组合成 ARENA ``RobotInterface``。

这样真机与仿真共用同一条闭环（``arena.client.ControlLoop``）：

    obs = robot.get_observation()          # 相机 + 关节状态
    chunk = client.infer(obs, instruction)  # 一次 VLA 推理
    for action in adapter.decode_chunk(chunk):
        robot.execute(action)               # 映射 + 低层下发

观测与动作的对接口径
------------------------------------------------------------------
* 观测 ``state``（proprio）：按 ``action_mode`` 组装，随后交给
  ``arena.adapter.EmbodimentAdapter`` 用 checkpoint 的 ``dataset_statistics.json``
  归一化 —— **顺序必须与训练数据一致**，这是最容易踩的静默错误；
* 动作：``execute`` 把单步动作交给 ``ActionMapper.map``，得到臂/夹爪目标后调
  ``controller.set_arm_targets``。

成功判定
------------------------------------------------------------------
真机抓取是否成功无法像仿真那样用 `objects_in_proximity` 自动判定。默认
``task_finished()`` 恒为 ``False``（由外部按步数/手动停止）；若你有视觉/夹爪
力反馈判据，请子类化并覆写该方法。
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

try:  # arena 属于本工作区；若缺失则降级为本地最小接口
    from arena.client import RobotInterface
except Exception:  # noqa: BLE001 - 独立运行时不强依赖 arena
    import abc

    class RobotInterface(abc.ABC):  # type: ignore[no-redef]
        """arena 未安装时的最小兼容接口占位。"""

        def reset(self):  # pragma: no cover - 占位
            raise NotImplementedError

        def get_observation(self):  # pragma: no cover
            raise NotImplementedError

        def execute(self, action):  # pragma: no cover
            raise NotImplementedError

from .action_mapper import ActionMapper
from .camera import CameraSource
from .g1_lowlevel import G1LowLevelController
from .joint_map import EE_ACTION_DIM

logger = logging.getLogger("g1d_deploy")


class G1RealRobot(RobotInterface):
    """G1-D 真机机器人（实现 ARENA ``RobotInterface``）。"""

    def __init__(
        self,
        controller: G1LowLevelController,
        action_mapper: ActionMapper,
        camera_head: CameraSource,
        camera_wrist: Optional[CameraSource] = None,
        action_mode: str = "joint",
        instruction: str = "pick up the object",
    ) -> None:
        """初始化。

        Args:
            controller: 低层关节控制器（已 ``connect`` + ``start``）。
            action_mapper: 动作映射器（关节空间或 EE 空间）。
            camera_head: 第三人称（头戴）相机。
            camera_wrist: 腕部相机；``None`` 表示该机型无腕部相机。
            action_mode: ``"joint"`` / ``"ee"``，决定 proprio 的组装口径。
            instruction: 默认任务描述。
        """
        self.controller = controller
        self.action_mapper = action_mapper
        self.camera_head = camera_head
        self.camera_wrist = camera_wrist
        self.action_mode = action_mode
        self.instruction = instruction
        self._running = True

    # ------------------------------------------------------------------
    # RobotInterface 实现
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, object]:
        """复位：保持当前姿态，返回首帧观测。"""
        self.controller.hold_current_pose()
        self._running = True
        return self.get_observation()

    def get_observation(self) -> Dict[str, object]:
        """采集相机帧 + 本体状态，组装成 ARENA 观测载荷。"""
        images: Dict[str, np.ndarray] = {}
        head = self.camera_head.read()
        if head is not None:
            images["head"] = head
        if self.camera_wrist is not None:
            wrist = self.camera_wrist.read()
            if wrist is not None:
                images["wrist"] = wrist
        return {"images": images, "state": self.build_state()}

    def build_state(self) -> np.ndarray:
        """按 ``action_mode`` 组装 proprio 向量。

        Returns:
            一维 ``float32`` proprio。关节空间 16 维；EE 空间 23 维占位。
        """
        arm = self.controller.get_arm_state()          # [左臂7, 右臂7] 实测
        gripper = self.controller.get_gripper_state()  # [左, 右] 实测位置

        if self.action_mode == "joint":
            # [左臂7, 右臂7, 左夹爪, 右夹爪] —— 与 joint_map 的默认动作顺序一致
            return np.concatenate([arm, gripper], axis=0).astype(np.float32)

        # EE 路径：checkpoint 的 proprio 通常含末端位姿/世界位姿等（见 DOC §12.3），
        # 真机很难凑齐，这里给出 23 维占位（臂 14 + 夹爪 2 + 零填充），并提示标定。
        # FIXME(标定): 按 EE checkpoint 的 dataset_statistics.json 重新确定顺序。
        logger.warning("EE 路径的 proprio 为占位实现，需按 checkpoint 标定")
        state = np.zeros(EE_ACTION_DIM, dtype=np.float32)
        state[:14] = arm
        state[14:16] = gripper
        return state

    def execute(self, action: np.ndarray) -> None:
        """执行单个动作：映射 → 低层下发目标。"""
        arm_state = self.controller.get_arm_state()
        targets = self.action_mapper.map(action, arm_state)
        self.controller.set_arm_targets(
            left_arm=targets["left_arm"],
            right_arm=targets["right_arm"],
            left_gripper=float(targets["left_gripper"]),
            right_gripper=float(targets["right_gripper"]),
        )

    def is_running(self) -> bool:
        return self._running and self.controller.is_running()

    def task_finished(self) -> bool:
        """真机默认无自动成功判定（由外部控制步数/手动停止）。"""
        return False

    def stop(self) -> None:
        """安全停止：急停低层控制并释放相机。"""
        self._running = False
        self.controller.stop()
        self.camera_head.close()
        if self.camera_wrist is not None:
            self.camera_wrist.close()
