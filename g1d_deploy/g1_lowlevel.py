"""G1-D 低层关节控制器：通过 DDS ``rt/lowcmd`` 以 500Hz 驱动手臂与夹爪。

这是真机抓取的**执行层**。它复用 SDK 官方低层例程的成熟套路，但按"只动上身、
下身保持"的安全姿态做了重构：

* 下半身（腿 + 腰，槽位 0..14）—— **保持接管瞬间的实测姿态**，不主动摆腿；
* 双臂（槽位 15..28）—— 由 VLA 动作映射得到的关节目标驱动；
* 夹爪/手部（槽位 29..30，可标定）—— 由夹爪命令驱动。

控制律与官方例程完全一致：每帧写 ``q / dq / tau / kp / kd / mode``，最后
``CRC().Crc(low_cmd)`` 重算校验码再 ``Write``。漏算 CRC 会被机器人整帧丢弃。

安全设计
------------------------------------------------------------------
* **接管即保持**：第一次收到 lowstate 后，把当前实测角作为所有槽位的初始目标，
  因此启动瞬间不会产生位置阶跃；
* **平滑过渡**：每个控制周期让"实际下发角"以有限步长逼近目标角（``max_step_rad``），
  避免目标突变导致抽动；
* **软限位**：臂关节动作先经 ``joint_map.clip_arm_action`` 裁剪；
* **看门狗**：超过 ``watchdog_s`` 没有新的动作目标，自动回到"保持当前姿态"；
* **急停**：``stop()`` 立即停止写线程并把下发角冻结在当前位置。

运行平台
------------------------------------------------------------------
低层控制线程使用 ``RecurrentThread``（Linux ``timerfd`` 定时），因此本模块
**只能在机器人上位机（Linux）上运行**；在 macOS/Windows 上仅能 import 做逻辑
自测，无法真正跑 500Hz 循环。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

import numpy as np

from .joint_map import (
    G1_NUM_MOTOR,
    HG_NUM_MOTOR,
    DEFAULT_KP,
    DEFAULT_KD,
    LEFT_ARM_SLOTS,
    RIGHT_ARM_SLOTS,
    GRIPPER_SLOTS,
    gripper_cmd_to_position,
    clip_arm_action,
)

logger = logging.getLogger("g1d_deploy")


class G1LowLevelController:
    """G1 低层 DDS 关节控制器（500Hz 写线程 + lowstate 订阅）。"""

    def __init__(
        self,
        network_interface: Optional[str] = None,
        domain_id: int = 0,
        control_dt: float = 0.002,
        max_step_rad: float = 0.02,
        watchdog_s: float = 2.0,
        kp: Optional[list] = None,
        kd: Optional[list] = None,
        dry_run: bool = False,
    ) -> None:
        """初始化（此时尚未建立 DDS 连接）。

        Args:
            network_interface: 网卡名（如 ``"eth0"``）；``None`` 交给 DDS 自动判定。
            domain_id: DDS 域 ID，必须与机器人一致，通常 0。
            control_dt: 控制周期（秒），默认 0.002 = 500Hz。
            max_step_rad: 每个周期单关节最大逼近步长（弧度），用于平滑。
            watchdog_s: 看门狗超时（秒），超时后回到"保持当前姿态"。
            kp/kd: 位置/速度增益（长度 29）；``None`` 用 :data:`DEFAULT_KP/KD`。
            dry_run: 为 True 时不真正连 DDS，用假 lowstate 跑控制律（仅逻辑自测）。
        """
        self.network_interface = network_interface
        self.domain_id = domain_id
        self.control_dt = control_dt
        self.max_step_rad = max_step_rad
        self.watchdog_s = watchdog_s
        self.dry_run = dry_run

        self.kp = np.asarray(kp if kp is not None else DEFAULT_KP, dtype=np.float32)
        self.kd = np.asarray(kd if kd is not None else DEFAULT_KD, dtype=np.float32)

        # 各槽位的目标角与"实际下发角"（长度取 HG 电机数 35，前 29 有效）
        self._q_target = np.zeros(HG_NUM_MOTOR, dtype=np.float32)
        self._q_cmd = np.zeros(HG_NUM_MOTOR, dtype=np.float32)
        self._target_lock = threading.Lock()

        # 状态
        self.low_state = None            # 最近一帧 LowState_
        self.mode_machine = 0
        self._got_state = False
        self._last_action_ts = 0.0
        self._running = False

        # DDS 对象（connect 后创建）
        self.lowcmd_publisher = None
        self.lowstate_subscriber = None
        self._write_thread = None

    # ------------------------------------------------------------------
    # 连接与启动
    # ------------------------------------------------------------------
    def connect(self, release_high_level: bool = True) -> None:
        """初始化 DDS 并建立 ``rt/lowcmd`` 发布与 ``rt/lowstate`` 订阅。

        顺序（与官方例程一致）：
        1. ``ChannelFactoryInitialize`` —— 建 Domain + DomainParticipant，失败会抛；
        2. （可选）``MotionSwitcherClient`` 循环 ``ReleaseMode`` 释放高层运控，
           否则低层写的 ``rt/lowcmd`` 会被高层丢弃；
        3. 建发布者 / 订阅者。

        Args:
            release_high_level: 是否先释放高层 sport 模式（真机必须为 True）。
        """
        if self.dry_run:
            logger.info("[dry-run] 跳过 DDS 连接，使用假 lowstate")
            self._seed_fake_state()
            self._got_state = True
            return

        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__LowCmd_,
        )

        ChannelFactoryInitialize(self.domain_id, self.network_interface)

        if release_high_level:
            self._release_high_level()

        self._low_cmd = unitree_hg_msg_dds__LowCmd_()

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self._on_low_state, queueLen=10)

    def start(self, timeout_s: float = 10.0) -> None:
        """等待收到第一帧 lowstate 后启动 500Hz 控制线程。

        Args:
            timeout_s: 等待首帧 lowstate 的最长时间（秒），超时抛 ``TimeoutError``。
        """
        if not self.dry_run:
            waited = 0.0
            while not self._got_state:
                if waited >= timeout_s:
                    raise TimeoutError("等待 rt/lowstate 首帧超时：请检查网卡/域/是否已释放高层模式")
                time.sleep(0.1)
                waited += 0.1

            # 接管瞬间：把当前实测角作为所有槽位的初始目标（避免位置阶跃）
            with self._target_lock:
                self._q_target[:G1_NUM_MOTOR] = np.asarray(
                    [m.q for m in self.low_state.motor_state[:G1_NUM_MOTOR]],
                    dtype=np.float32,
                )
                self._q_cmd[:] = self._q_target

        self._running = True
        self._last_action_ts = time.time()

        if self.dry_run:
            return

        from unitree_sdk2py.utils.thread import RecurrentThread

        self._write_thread = RecurrentThread(
            interval=self.control_dt, target=self._write_loop, name="g1_lowlevel"
        )
        self._write_thread.Start()
        logger.info("✓ G1 低层控制线程已启动 (%.1f Hz)", 1.0 / self.control_dt)

    # ------------------------------------------------------------------
    # 对外控制接口（由上层闭环调用，线程安全）
    # ------------------------------------------------------------------
    def set_arm_targets(
        self,
        left_arm: np.ndarray,
        right_arm: np.ndarray,
        left_gripper: float = 0.0,
        right_gripper: float = 0.0,
    ) -> None:
        """设置双臂 + 夹爪的目标关节角。

        本方法只更新**目标**，真正的平滑逼近发生在 500Hz 写线程里。
        臂动作会先经软限位裁剪。

        Args:
            left_arm: 左臂 7 关节目标（弧度）。
            right_arm: 右臂 7 关节目标（弧度）。
            left_gripper/right_gripper: 夹爪命令（约定 0=张开、1=闭合）。
        """
        left_arm = clip_arm_action(np.asarray(left_arm, dtype=np.float32).reshape(-1))
        right_arm = clip_arm_action(np.asarray(right_arm, dtype=np.float32).reshape(-1))
        if left_arm.shape[0] != 7 or right_arm.shape[0] != 7:
            raise ValueError("left_arm / right_arm 必须各为 7 维")

        with self._target_lock:
            for slot, q in zip(LEFT_ARM_SLOTS, left_arm):
                self._q_target[slot] = float(q)
            for slot, q in zip(RIGHT_ARM_SLOTS, right_arm):
                self._q_target[slot] = float(q)
            for name, slot in GRIPPER_SLOTS.items():
                if name == "left":
                    self._q_target[slot] = gripper_cmd_to_position(left_gripper)
                elif name == "right":
                    self._q_target[slot] = gripper_cmd_to_position(right_gripper)
        self._last_action_ts = time.time()

    def hold_current_pose(self) -> None:
        """把当前实测姿态设为所有槽位的目标（用于急停后的恢复/保持）。"""
        if self.low_state is None:
            return
        with self._target_lock:
            self._q_target[:G1_NUM_MOTOR] = np.asarray(
                [m.q for m in self.low_state.motor_state[:G1_NUM_MOTOR]],
                dtype=np.float32,
            )
        self._last_action_ts = time.time()

    def get_arm_state(self) -> np.ndarray:
        """返回双臂 14 维实测关节角 ``[左臂7, 右臂7]``。

        Returns:
            实测角数组；未收到 lowstate 时返回全 0。
        """
        if self.low_state is None:
            return np.zeros(14, dtype=np.float32)
        slots = LEFT_ARM_SLOTS + RIGHT_ARM_SLOTS
        return np.asarray(
            [self.low_state.motor_state[i].q for i in slots], dtype=np.float32
        )

    def get_gripper_state(self) -> np.ndarray:
        """返回夹爪实测位置 ``[左, 右]``（依据 :data:`GRIPPER_SLOTS`）。"""
        if self.low_state is None:
            return np.zeros(2, dtype=np.float32)
        return np.asarray(
            [self.low_state.motor_state[GRIPPER_SLOTS[n]].q for n in ("left", "right")],
            dtype=np.float32,
        )

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        """急停：停止写线程并冻结当前下发角（保持现状，不再逼近新目标）。"""
        self._running = False
        if self._write_thread is not None:
            # RecurrentThread.Wait 会置 quit 并等待线程退出
            try:
                self._write_thread.Wait(timeout=1.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("写线程退出异常（可忽略）: %s", exc)
        logger.info("⚠ G1 低层控制已急停")

    # ------------------------------------------------------------------
    # 内部：订阅回调 + 写循环
    # ------------------------------------------------------------------
    def _on_low_state(self, msg) -> None:
        """``rt/lowstate`` 订阅回调（运行在 SDK 队列消费线程里）。

        只做轻量赋值：缓存状态、记录 mode_machine、置首帧标记。绝不做耗时操作。
        """
        self.low_state = msg
        if not self._got_state:
            self.mode_machine = msg.mode_machine
            self._got_state = True

    def _write_loop(self) -> None:
        """500Hz 控制律主体（运行在 RecurrentThread 里）。

        每个周期：
        1. 看门狗：太久没收到新动作目标，就回到"保持当前姿态"；
        2. 平滑：把实际下发角 ``_q_cmd`` 以 ``max_step_rad`` 逼近 ``_q_target``；
        3. 逐槽位写 ``mode/q/dq/tau/kp/kd``；
        4. 写 ``mode_pr / mode_machine``；
        5. 算 CRC 并 ``Write``。
        """
        if self.low_state is None:
            return

        now = time.time()
        if now - self._last_action_ts > self.watchdog_s:
            # 看门狗触发：回退到保持当前姿态（等价于 hold_current_pose 但不发动作）
            with self._target_lock:
                self._q_target[:G1_NUM_MOTOR] = np.asarray(
                    [m.q for m in self.low_state.motor_state[:G1_NUM_MOTOR]],
                    dtype=np.float32,
                )

        with self._target_lock:
            target = self._q_target.copy()
        cmd = self._q_cmd

        # 平滑逼近：每个关节每周期最多走 max_step_rad
        delta = target - cmd
        step = np.clip(delta, -self.max_step_rad, self.max_step_rad)
        cmd[:] = cmd + step

        mode_pr = 0  # PR 串联模式（臂关节默认）
        mode_machine = self.mode_machine

        for i in range(G1_NUM_MOTOR):
            mc = self._low_cmd.motor_cmd[i]
            mc.mode = 1              # 1: 使能
            mc.tau = 0.0
            mc.q = float(cmd[i])
            mc.dq = 0.0
            mc.kp = float(self.kp[i])
            mc.kd = float(self.kd[i])

        # 夹爪槽位（29..34 预留区）也一并下发，kp/kd 用较小值
        for name, slot in GRIPPER_SLOTS.items():
            if slot < G1_NUM_MOTOR or slot >= HG_NUM_MOTOR:
                continue
            mc = self._low_cmd.motor_cmd[slot]
            mc.mode = 1
            mc.tau = 0.0
            mc.q = float(cmd[slot])
            mc.dq = 0.0
            mc.kp = 40.0
            mc.kd = 1.0

        self._low_cmd.mode_pr = mode_pr
        self._low_cmd.mode_machine = mode_machine

        from unitree_sdk2py.utils.crc import CRC

        self._low_cmd.crc = CRC().Crc(self._low_cmd)
        self.lowcmd_publisher.Write(self._low_cmd)

    def _release_high_level(self) -> None:
        """用 MotionSwitcherClient 释放高层运控模式，直到没有模式占用控制权。"""
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )

        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()

        status, result = msc.CheckMode()
        while result.get("name"):
            logger.info("释放高层模式: %s", result.get("name"))
            msc.ReleaseMode()
            status, result = msc.CheckMode()
            time.sleep(1)
        logger.info("✓ 高层运控模式已释放")

    def _seed_fake_state(self) -> None:
        """dry-run 用：构造一个最小可用的假 lowstate（全 0 关节角）。"""
        class _Motor:  # noqa: D401 - 最小假对象
            q = 0.0

        class _State:
            mode_machine = 0
            motor_state = [_Motor() for _ in range(HG_NUM_MOTOR)]

        self.low_state = _State()
        self.mode_machine = 0
