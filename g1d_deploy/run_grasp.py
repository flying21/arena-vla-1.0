#!/usr/bin/env python3
"""G1-D 真机 VLA 闭环抓取入口。

把本工作区的 VLA 策略服务器、ARENA 适配器与 g1d_deploy 低层执行链路串起来：

    VLA 服务器 (--server-url, GPU)
        ▲  POST /act（观测）
        │
    G1RealRobot (本包) ── 相机 + lowstate 关节状态 ──▶ 观测
        │
        └── 动作 → action_mapper → 500Hz 低层关节控制 ──▶ 机器人

运行前提（务必先看 README.md 的检查表）
------------------------------------------------------------------
1. 机器人已开机，本机与机器人同一 DDS 域、网卡已指定；
2. 已先跑通 ``smoke_test.py`` 验证低层链路；
3. VLA 策略服务器已在 GPU 上运行且 ``/health`` 返回 200；
4. checkpoint 的 ``dataset_statistics.json`` 路径已知（用于动作/状态反归一化）。

运行示例
------------------------------------------------------------------
    python g1d_deploy/run_grasp.py \
        --interface eth0 \
        --server-url http://127.0.0.1:8777/act \
        --instruction "pick up the brown box and place it in the bin" \
        --norm-stats /path/to/checkpoint/dataset_statistics.json \
        --unnorm-key new_embodiment \
        --action-mode joint \
        --max-steps 200
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from .g1_lowlevel import G1LowLevelController
from .action_mapper import build_action_mapper
from .camera import build_camera
from .g1_robot import G1RealRobot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("g1d_deploy.run_grasp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1-D 真机 VLA 闭环抓取")
    # 机器人连接
    p.add_argument("--interface", type=str, default=None, help="网卡名（如 eth0）")
    p.add_argument("--domain", type=int, default=0, help="DDS 域 ID")
    p.add_argument("--dry-run", action="store_true", help="不连 DDS/相机，仅逻辑自测")
    # VLA 服务
    p.add_argument("--server-url", type=str, default="http://127.0.0.1:8777/act")
    p.add_argument("--instruction", type=str, default="pick up the object")
    # 动作口径与归一化
    p.add_argument("--action-mode", type=str, default="joint", choices=["joint", "ee"])
    p.add_argument("--norm-stats", type=str, default="", help="dataset_statistics.json 路径")
    p.add_argument("--unnorm-key", type=str, default="new_embodiment")
    p.add_argument("--norm-type", type=str, default="bounds",
                   choices=["normal", "bounds", "bounds_q99"])
    # 相机
    p.add_argument("--head-camera", type=str, default="webcam", help="mock|webcam")
    p.add_argument("--head-camera-src", type=str, default="0", help="webcam 索引或 RTSP 地址")
    p.add_argument("--wrist-camera", type=str, default="", help="腕部相机：mock|webcam（空=无）")
    # 闭环参数
    p.add_argument("--max-steps", type=int, default=200, help="闭环最大步数")
    p.add_argument("--step-dt", type=float, default=0.05, help="执行动作块内每步间隔（秒）")
    return p.parse_args()


def _load_norm_stats(path: str, unnorm_key: str):
    """加载 checkpoint 统计量，取指定任务键的 ``{action, proprio}``。"""
    from arena.config import load_norm_stats

    full = load_norm_stats(path)
    if unnorm_key not in full:
        logger.warning("unnorm_key=%s 不在统计量键中: %s，将使用第一个可用键",
                       unnorm_key, sorted(full))
        unnorm_key = next(iter(full))
    stats = full[unnorm_key]
    logger.info("✓ 已加载归一化统计量（任务键=%s）", unnorm_key)
    return stats


def _head_source(src: str):
    """把 --head-camera-src 字符串转成 int 或保留 str（RTSP/HTTP）。"""
    return int(src) if src.lstrip("-").isdigit() else src


def main() -> int:
    args = parse_args()

    # ---- 低层控制器 + 相机 + 映射 + 真机适配 ---------------------------
    ctrl = G1LowLevelController(
        network_interface=args.interface, domain_id=args.domain, dry_run=args.dry_run
    )
    ctrl.connect(release_high_level=not args.dry_run)
    ctrl.start(timeout_s=10.0)

    def _make_camera(kind: str, src):
        """按 kind 构造相机；webcam 需要 source 参数，mock 不需要。"""
        if kind == "webcam":
            return build_camera("webcam", source=_head_source(src))
        return build_camera("mock")

    head_cam = _make_camera(args.head_camera, args.head_camera_src)
    wrist_cam = _make_camera(args.wrist_camera, "0") if args.wrist_camera else None

    mapper = build_action_mapper(args.action_mode)
    robot = G1RealRobot(
        controller=ctrl,
        action_mapper=mapper,
        camera_head=head_cam,
        camera_wrist=wrist_cam,
        action_mode=args.action_mode,
        instruction=args.instruction,
    )

    # ---- ARENA 适配器 + 策略客户端 ------------------------------------
    from arena.adapter import EmbodimentAdapter
    from arena.client import PolicyClient
    from arena.config import AdapterConfig, ClientConfig, NormalizationType

    norm_stats = {}
    if args.norm_stats:
        norm_stats = _load_norm_stats(args.norm_stats, args.unnorm_key)
    else:
        logger.warning("⚠ 未提供 --norm-stats：状态/动作将不归一化（仅适合干跑，真机必须提供）")

    adapter = EmbodimentAdapter(
        AdapterConfig(
            robot_type="unitree_g1",
            normalization_type=NormalizationType(args.norm_type),
        ),
        norm_stats=norm_stats,
        instruction=args.instruction,
    )
    client = PolicyClient(ClientConfig(server_url=args.server_url))

    # ---- 闭环 ----------------------------------------------------------
    logger.info("▶ 开始闭环抓取 (instruction=%r, max_steps=%d)", args.instruction, args.max_steps)
    robot.reset()
    try:
        for step in range(args.max_steps):
            if not robot.is_running():
                break
            obs = robot.get_observation()
            obs.setdefault("instruction", args.instruction)
            observation = adapter.encode_observation(obs)
            chunk = client.infer(observation, instruction=args.instruction)
            decoded = adapter.decode_chunk(chunk.actions)
            for action in decoded:
                robot.execute(action)
                time.sleep(args.step_dt)
                if not robot.is_running():
                    break
            if step % 20 == 0:
                logger.info("step %d/%d, 臂状态=%s", step, args.max_steps,
                            np.round(ctrl.get_arm_state(), 3))
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，安全停止")
    finally:
        robot.stop()

    logger.info("✓ 闭环结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
