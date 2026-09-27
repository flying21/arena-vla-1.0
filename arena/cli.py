# 麻雀虽小智能科技（武汉）有限公司
"""ARENA 命令行入口。

三个子命令与技术报告的工作流一一对应：

* ``arena server`` —— 启动 VLA 策略服务器（第 2.3 节）
* ``arena sim``    —— 在 Isaac Lab / Arena 仿真中跑闭环（第 3.2 节）
* ``arena real``   —— 在 Unitree 真机上跑闭环（第 3.3 节）

设计原则：**重依赖全部惰性导入**。
    ``lerobot``（仿真）、``unitree_deploy``（真机）、``fastapi``（服务器）
    都只在真正执行对应子命令时才导入。这样：
    ``arena --help`` 在裸环境里也能用；某个依赖缺失只会影响它自己的子命令，
    而不会让整个 CLI 失效。

进程内 vs HTTP 两种推理路径
------------------------------------------------------------------
``arena sim`` / ``arena real`` 支持两种客户端：

* **进程内**（``--inprocess`` 或 ``--backend mock``）：CLI 进程里直接构造后端并调用，
  没有网络往返。适合本地调试、冒烟测试、无服务器场景。
* **HTTP**：通过 :class:`arena.client.PolicyClient` 访问远端 ``/act``。
  这才是"模型服务化"的正常部署形态（GPU 机器跑服务器，机器人侧跑客户端）。

两者对控制回路透明 —— :class:`~arena.client.ControlLoop` 只要求对象有
``infer(observation, instruction)`` 方法，因此 :class:`_InProcessClient`
可以无缝顶替 :class:`~arena.client.PolicyClient`。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List

import numpy as np

from arena.config import ServerConfig
from arena.server import serve

# 统一日志格式：时间 [级别] 消息
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# 子命令：server
# ---------------------------------------------------------------------------


def _server_command(args: argparse.Namespace) -> int:
    """把命令行参数组装成 :class:`ServerConfig` 并启动服务器（阻塞）。

    注意两个取反参数：``--no_bf16`` / ``--no_center_crop`` 是"关闭开关"，
    因此这里用 ``not`` 转回正向语义。

    Args:
        args: argparse 解析出的参数命名空间。

    Returns:
        进程退出码（正常情况不会返回，因为 :func:`serve` 会阻塞）。
    """
    config = ServerConfig(
        backend=args.backend,
        host=args.host,
        port=args.port,
        ckpt_path=args.ckpt_path,
        vlm_pretrained_path=args.vlm_pretrained_path,
        unnorm_key=args.unnorm_key,
        use_bf16=not args.no_bf16,
        center_crop=not args.no_center_crop,
        device=args.device,
        upstream_url=args.upstream_url,
    )
    serve(config)
    return 0


# ---------------------------------------------------------------------------
# 子命令：sim / real（共用同一套闭环逻辑）
# ---------------------------------------------------------------------------


def _make_inprocess_client(backend: str, ckpt_path: str, vlm_pretrained_path, unnorm_key: str):
    """构造一个进程内推理函数 ``infer(observation, instruction) -> ndarray``。

    与 HTTP 客户端的关键差别：**不做任何序列化**。直接把
    :meth:`Observation.to_dict` 的结果喂给后端，因此完全没有传输开销，
    也不会踩 numpy/JSON 编解码的坑。

    Args:
        backend: 后端名（mock / unifolm_vla / openpi / http）。
        ckpt_path: ``unifolm_vla`` 后端的 checkpoint 路径。
        vlm_pretrained_path: 可选的 VLM 主干路径覆盖。
        unnorm_key: 反归一化使用的数据集键。

    Returns:
        可直接调用的 ``infer`` 函数。
    """
    from arena.backends import build_backend
    from arena.types import Observation

    backend_config = ServerConfig(
        backend=backend,
        ckpt_path=ckpt_path,
        vlm_pretrained_path=vlm_pretrained_path,
        unnorm_key=unnorm_key,
    )
    policy = build_backend(backend_config)

    def infer(observation: Observation, instruction: str) -> np.ndarray:
        """在进程内直接执行一次推理。"""
        return policy.infer([observation.to_dict()], instruction)

    return infer


class _InProcessClient:
    """把本地后端包装成 :class:`~arena.client.PolicyClient` 的接口形状。

    它只需实现 ``infer(observation, instruction) -> ActionChunk``，
    就能被 :class:`~arena.client.ControlLoop` 接受 —— 这正是"控制回路只依赖
    鸭子类型协议"带来的好处。
    """

    def __init__(self, infer_fn, instruction: str) -> None:
        """保存推理函数与默认指令。"""
        self._infer_fn = infer_fn
        self._instruction = instruction

    def infer(self, observation, instruction=None):
        """执行推理并包装成 :class:`~arena.types.ActionChunk`。"""
        from arena.types import ActionChunk

        actions = self._infer_fn(observation, instruction or self._instruction)
        return ActionChunk(actions=actions)


def _run_loop(args: argparse.Namespace) -> int:
    """``arena sim`` / ``arena real`` 的共同实现。

    流程：

    1. 构造适配器（带上默认任务指令）；
    2. 选择客户端：进程内（mock / ``--inprocess``）或 HTTP；
    3. 按 ``args.mode`` 选择具身：仿真环境（:class:`ArenaSimRobot`）
       或真机 SDK（:class:`UnitreeRobot`）；
    4. 运行 :class:`~arena.client.ControlLoop` 并打印 JSON 结果。

    Args:
        args: argparse 参数命名空间（含 ``mode`` 字段）。

    Returns:
        退出码 0。
    """
    from arena.adapter import EmbodimentAdapter
    from arena.client import ArenaSimRobot, ControlLoop, PolicyClient, UnitreeRobot
    from arena.config import AdapterConfig, ClientConfig

    adapter = EmbodimentAdapter(AdapterConfig(robot_type=args.robot), instruction=args.instruction)

    if args.inprocess or args.backend == "mock":
        # 进程内路径：无需服务器；mock 后端天然只能进程内使用
        infer_fn = _make_inprocess_client(
            args.backend, args.ckpt_path, args.vlm_pretrained_path, args.unnorm_key
        )
        client = _InProcessClient(infer_fn, args.instruction)
    else:
        # HTTP 路径：走真实的服务化部署
        client = PolicyClient(ClientConfig(server_url=args.server_url))

    if args.mode == "sim":
        env = _build_arena_env(args)
        robot = ArenaSimRobot(env, instruction=args.instruction, max_steps=args.max_steps)
    else:
        sdk = _build_unitree_sdk(args)
        robot = UnitreeRobot(sdk, instruction=args.instruction)

    loop = ControlLoop(robot, client, adapter, instruction=args.instruction, max_episodes=args.episodes)
    results: List[Dict[str, Any]] = loop.run(args.instruction)
    print(json.dumps(results, indent=2))
    return 0


def _build_arena_env(args: argparse.Namespace):
    """通过 LeRobot 构造一个 Isaac Lab / Arena 环境。

    LeRobot 的 ``make_env`` 返回 ``{任务名: [环境实例, ...]}``，
    这里取单环境（``n_envs=1``）的第一个实例。

    Args:
        args: 需要 ``env_id``（Hub 仓库）与 ``env_task``（任务名）。

    Returns:
        单个 gymnasium 兼容环境。
    """
    from lerobot.envs.factory import make_env

    envs = make_env(args.env_id, n_envs=1, trust_remote_code=True)
    return envs[args.env_task][0]


def _build_unitree_sdk(args: argparse.Namespace):
    """构造宇树部署控制器（惰性导入，真机环境才需要）。

    Args:
        args: 需要 ``robot_host``。

    Returns:
        ``unitree_deploy`` 客户端实例。
    """
    from unitree_deploy.robot_client import UnitreeDeployClient  # type: ignore

    return UnitreeDeployClient(server_host=args.robot_host)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """构造完整的 argparse 解析器（三个子命令）。

    Returns:
        配置好的 :class:`argparse.ArgumentParser`。
    """
    parser = argparse.ArgumentParser(prog="arena", description="ARENA VLA Server-Client-Adapter")
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- arena server ----
    server = sub.add_parser("server", help="Launch the VLA policy server")
    server.add_argument("--backend", default="mock", choices=["mock", "unifolm_vla", "openpi", "http"])
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8777)
    server.add_argument("--ckpt_path", default="")
    server.add_argument("--vlm_pretrained_path", default=None)
    server.add_argument("--unnorm_key", default="new_embodiment")
    server.add_argument("--device", default="cuda:0")
    server.add_argument("--upstream_url", default=None)
    server.add_argument("--no_bf16", action="store_true", help="关闭 bfloat16 推理")
    server.add_argument("--no_center_crop", action="store_true", help="关闭图像中心裁剪")
    server.set_defaults(func=_server_command)

    # ---- arena sim / arena real（共用一套参数）----
    for name, mode in (("sim", "sim"), ("real", "real")):
        run = sub.add_parser(name, help=f"Run the closed loop in {mode} mode")
        run.add_argument("--instruction", default="pick up the cube")
        run.add_argument("--robot", default="unitree_g1")
        run.add_argument("--backend", default="mock", choices=["mock", "unifolm_vla", "openpi", "http"])
        run.add_argument("--ckpt_path", default="")
        run.add_argument("--vlm_pretrained_path", default=None)
        run.add_argument("--unnorm_key", default="new_embodiment")
        run.add_argument("--server_url", default="http://127.0.0.1:8777/act")
        run.add_argument("--inprocess", action="store_true", help="强制使用进程内后端（不走 HTTP）")
        run.add_argument("--episodes", type=int, default=1)
        run.add_argument("--max_steps", type=int, default=300)
        run.add_argument("--env_id", default="nvkartik/isaaclab-arena-envs")
        run.add_argument("--env_task", default="gr1_microwave")
        run.add_argument("--robot_host", default="127.0.0.1")
        # mode 由循环注入，避免在 _run_loop 里再判断子命令名
        run.set_defaults(func=_run_loop, mode=mode)

    return parser


def main(argv=None) -> int:
    """CLI 主入口。

    Args:
        argv: 参数列表；``None`` 时读取 ``sys.argv``（便于测试时注入）。

    Returns:
        子命令返回的退出码。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
