# 麻雀虽小智能科技（武汉）有限公司
"""ARENA 端到端演示：无需 GPU、无需 Isaac Lab、无需网络。

数据流::

    Observation -> Adapter -> Policy Backend -> Action Chunk -> Robot

用途
------------------------------------------------------------------
这是**验证框架自身是否可用**的最小闭环。它把整条链路上所有"外部依赖"
都替换成了本地替身：

======================  ==================================================
真实组件                 本 demo 中的替身
======================  ==================================================
Isaac Lab / 真机          :class:`FakeRobot`（跑满固定步数即上报成功）
GPU 上的 VLA 模型         ``mock`` 后端（确定性动作，见 arena/backends.py）
HTTP 服务器               :class:`InProcessClient`（进程内直接调后端）
======================  ==================================================

因此它非常适合：新环境自检、CI 冒烟测试、阅读代码时的"最小可运行样例"。

用法::

    python demo.py
    python demo.py --episodes 2 --sim2real-level 3

注意 ``--sim2real-level`` 会真实生效：level >= 2 时观测图像会被施加
亮度/对比度/相机扰动，动作会被施加延迟（见 arena/sim2real.py）。
"""

from __future__ import annotations

import argparse

import numpy as np

from arena.adapter import EmbodimentAdapter
from arena.backends import build_backend
from arena.client import RobotInterface
from arena.config import AdapterConfig, ServerConfig
from arena.sim2real import Sim2RealCurriculum
from arena.types import ActionChunk


class FakeRobot(RobotInterface):
    """脚本化的假机器人：跑满 ``horizon`` 步后即判定任务成功。

    它不模拟任何物理，只保证：

    * 提供两个相机（用渐变图填充，便于肉眼确认图像确实被处理过）；
    * 维护一个 8 维"状态"，并把收到的动作回写成新状态（形成反馈回路，
      使后续观测随动作变化，而不是一直返回同一个零向量）。
    """

    def __init__(self, horizon: int = 12, camera_size: int = 256) -> None:
        """初始化。

        Args:
            horizon: 经过多少步后判定成功。
            camera_size: 假相机画面的边长（像素）。
        """
        self.horizon = horizon
        self.camera_size = camera_size
        self.steps = 0
        self.state = np.zeros(8, dtype=np.float32)
        self.executed = []  # 记录已执行的动作，便于调试/断言

    def _frame(self) -> np.ndarray:
        """生成一帧确定性的渐变测试图（水平灰度渐变转 RGB）。

        用渐变而非纯色，是为了让"图像是否被缩放/扰动"这类问题肉眼可辨。
        """
        gradient = np.linspace(0, 255, self.camera_size, dtype=np.uint8)
        return np.tile(gradient[None, :, None], (self.camera_size, 1, 3))

    def reset(self):
        """复位步数、状态与动作记录，并返回首帧观测。"""
        self.steps = 0
        self.state = np.zeros(8, dtype=np.float32)
        self.executed = []
        return self.get_observation()

    def get_observation(self):
        """返回 ``{"images": {"head", "wrist"}, "state"}``（ARENA 规范载荷）。

        ``state.copy()`` 是必要的：否则下游若原地修改状态会污染机器人内部值。
        """
        return {
            "images": {"head": self._frame(), "wrist": self._frame()},
            "state": self.state.copy(),
        }

    def execute(self, action):
        """执行单个动作：记录它，并把动作的前 8 维当作新状态（闭环反馈）。"""
        self.executed.append(np.asarray(action, dtype=np.float32))
        self.state = np.asarray(action, dtype=np.float32)[: self.state.shape[0]]
        self.steps += 1

    def is_running(self):
        """未满 ``horizon`` 步时继续运行。"""
        return self.steps < self.horizon

    def task_finished(self):
        """步数达到 ``horizon`` 即视为任务成功。"""
        return self.steps >= self.horizon


class InProcessClient:
    """进程内策略客户端：直接调用后端，完全没有 HTTP 往返。

    只需实现 ``infer(observation, instruction)`` 就能被 ``ControlLoop`` 使用 ——
    这里刻意不继承 ``PolicyClient``，以演示控制回路依赖的是**协议**而非具体类。
    """

    def __init__(self, backend) -> None:
        """保存后端实例。"""
        self.backend = backend

    def infer(self, observation, instruction=None):
        """把观测序列化后交给后端，并包装成 :class:`ActionChunk` 返回。"""
        actions = self.backend.infer([observation.to_dict()], instruction or "")
        return ActionChunk(actions=actions)


def main() -> None:
    """跑 ``--episodes`` 个 episode，打印每轮的步数/块数/动作维度摘要。"""
    parser = argparse.ArgumentParser(description="ARENA end-to-end demo")
    parser.add_argument("--backend", default="mock", choices=["mock", "unifolm_vla", "openpi"])
    parser.add_argument("--instruction", default="pick up the red cube and place it in the bowl")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--sim2real-level", type=int, default=1, help="域随机化等级 1..4")
    args = parser.parse_args()

    print("=" * 72)
    print("ARENA 0.1.0 - VLA Server-Client-Adapter demo")
    print("=" * 72)

    # 组装四件套：适配器、后端、客户端、机器人
    adapter = EmbodimentAdapter(AdapterConfig(), instruction=args.instruction)
    backend = build_backend(ServerConfig(backend=args.backend))
    client = InProcessClient(backend)
    robot = FakeRobot()

    # 域随机化课程（默认 1 级 = 严格恒等，不改变任何数据）
    curriculum = Sim2RealCurriculum()
    curriculum.reset(args.sim2real_level)
    print("Sim-to-real profile: %s" % curriculum.profile.describe())

    for episode in range(args.episodes):
        robot.reset()
        chunk_index = 0
        chunk = None  # 仅用于最后打印 action_dim，保证至少被赋值一次
        while robot.is_running() and not robot.task_finished():
            # ① 取观测 -> ② 施加域随机化 -> ③ 适配器编码成规范观测
            obs = curriculum.randomize_observation(robot.get_observation())
            obs["instruction"] = args.instruction
            encoded = adapter.encode_observation(obs)
            # ④ 推理取回动作块 -> ⑤ 解码 -> ⑥ 逐步执行（带延迟随机化）
            chunk = client.infer(encoded, instruction=args.instruction)
            for action in adapter.decode_chunk(chunk.actions):
                robot.execute(curriculum.delay_action(action))
                if robot.task_finished():
                    break  # 成功即停，不再执行块内剩余动作
            chunk_index += 1
        print(
            "Episode %d: success=%s steps=%d chunks=%d action_dim=%d"
            % (episode + 1, robot.task_finished(), robot.steps, chunk_index, chunk.action_dim)
        )
    print("=" * 72)
    print("Demo complete. Pipeline: Adapter -> Server -> Adapter -> Robot.")


if __name__ == "__main__":
    main()
