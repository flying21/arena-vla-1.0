#!/usr/bin/env python3
"""G1-D 低层链路冒烟测试（**不接 VLA**，先验证"能不能安全驱动手臂"）。

这是真机第一次上电时**必须最先跑**的脚本，目的只有一个：确认
"上位机 → DDS → 机器人低层" 这条链路是通的、CRC 是合法的、手臂能按目标角
安全地小幅运动。**不要**先跑 VLA 闭环 —— 链路都没验证就上策略是危险的。

测试流程（逐步增强，均可安全中断）
------------------------------------------------------------------
1. 建立 DDS、释放高层模式、订阅 lowstate、启动 500Hz 写线程；
2. 保持当前姿态 2 秒（验证能持续收到 lowstate、写帧不报错）；
3. （可选 ``--move``）双臂肩关节做 ±10° 的**缓慢**正弦摆动，验证关节确实在动、
   方向是否符合预期；
4. 打印双臂实测角与夹爪位置，Ctrl+C 急停退出。

运行示例
------------------------------------------------------------------
    # 真机（eth0 网卡，小幅运动）
    python g1d_deploy/smoke_test.py --interface eth0 --move

    # 无硬件：只验证控制律/映射逻辑（不连 DDS）
    python g1d_deploy/smoke_test.py --dry-run --move
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from .g1_lowlevel import G1LowLevelController
from .action_mapper import JointSpaceMapper
from .camera import build_camera
from .g1_robot import G1RealRobot

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("g1d_deploy.smoke_test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G1-D 低层链路冒烟测试")
    p.add_argument("--interface", type=str, default=None, help="网卡名（如 eth0）")
    p.add_argument("--domain", type=int, default=0, help="DDS 域 ID")
    p.add_argument("--dry-run", action="store_true", help="不连 DDS，仅逻辑自测")
    p.add_argument("--move", action="store_true", help="是否做小幅手臂摆动")
    p.add_argument("--hold-s", type=float, default=2.0, help="启动后保持姿态的秒数")
    p.add_argument("--swing-s", type=float, default=6.0, help="摆动测试持续秒数")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("▶ 构造低层控制器（dry_run=%s）", args.dry_run)
    ctrl = G1LowLevelController(
        network_interface=args.interface,
        domain_id=args.domain,
        dry_run=args.dry_run,
    )
    ctrl.connect(release_high_level=not args.dry_run)
    ctrl.start(timeout_s=10.0)

    # 用 mock 相机 + 关节空间映射，构造最小可用的 G1RealRobot（仅用其观测/状态接口）
    robot = G1RealRobot(
        controller=ctrl,
        action_mapper=JointSpaceMapper(),
        camera_head=build_camera("mock"),
        action_mode="joint",
        instruction="smoke test",
    )

    logger.info("✓ 链路就绪。保持当前姿态 %.1f 秒（请观察机器人是否稳定）...", args.hold_s)
    time.sleep(args.hold_s)
    logger.info("双臂实测角: %s", np.round(robot.build_state(), 3))

    if args.move:
        logger.info("▶ 小幅摆臂测试（肩 pitch ±10°，周期约 4s）...")
        # 以当前实测角为基准，只在肩 pitch 上叠加小幅度正弦
        base = ctrl.get_arm_state().copy()  # [左臂7, 右臂7]
        t0 = time.time()
        try:
            while time.time() - t0 < args.swing_s:
                t = time.time() - t0
                amp = np.deg2rad(10.0)
                left = base[0:7].copy()
                right = base[7:14].copy()
                left[0] = base[0] + amp * np.sin(2 * np.pi * t / 4.0)   # 左肩 pitch
                right[0] = base[7] + amp * np.sin(2 * np.pi * t / 4.0)  # 右肩 pitch
                ctrl.set_arm_targets(left, right, left_gripper=0.0, right_gripper=0.0)
                time.sleep(0.05)
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C，停止摆动")
    else:
        logger.info("（跳过摆臂测试，加 --move 可启用）")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C")

    logger.info("▶ 急停并退出")
    robot.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
