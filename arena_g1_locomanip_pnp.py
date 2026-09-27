#!/usr/bin/env python3
"""麻雀虽小智能科技（武汉）有限公司
══════════════════════════════════════════════════════════════════════════
  ARENA 0.1.0 — G1 人形机器人 | 移动操作箱子抓取与放置 (Loco-Manipulation PnP)
══════════════════════════════════════════════════════════════════════════

任务描述
════════
G1 移动操作箱子抓取与放置任务 (G1 Loco-Manipulation Box Pick and Place Task)
是指宇树科技 G1 人形机器人结合下肢移动 (Locomotion) 与双臂操作 (Manipulation)
来完成箱子抓取并放置到目标分拣箱的任务。

核心词汇:
  - G1: 宇树科技 (Unitree) 研发的通用人形机器人
  - Loco-Manipulation (移动操作): 机器人边移动边进行手臂抓取和操作
  - Pick and Place (抓取与放置): 识别、夹取物体并放置到目标位置

任务流程 (4 个阶段):
  ┌─────────────────────────────────────────────────────────────────┐
  │ 阶段 1: 导航至操作台 (navigate_to_table)                        │
  │   G1 从初始位置步行移动到放有棕色箱子的操作台前                     │
  │   导航子目标: [0.18, 0.18, 0.0] → 直行 18cm 到箱子面前            │
  ├─────────────────────────────────────────────────────────────────┤
  │ 阶段 2: 转向操作台 (turn_in_place)                               │
  │   G1 在操作台面前原地转身 1.78 rad (~102°) 面朝箱子                 │
  │   导航子目标: [0.18, 0.18, -1.78] → 保持位置，旋转朝向箱子          │
  ├─────────────────────────────────────────────────────────────────┤
  │ 阶段 3: 导航至分拣箱 (navigate_to_bin)                           │
  │   G1 抓住箱子后步行移动到蓝色分拣箱位置                              │
  │   导航子目标: [-0.0955, -1.107, -1.78] → 移动 1.3m 到分拣箱前       │
  ├─────────────────────────────────────────────────────────────────┤
  │ 阶段 4: 放置箱子 + 成功检测                                       │
  │   G1 松开箱子，检测箱子是否在分拣箱范围内                            │
  │   成功条件: 箱子在分拣箱 26×13×15cm 接近范围内                       │
  └─────────────────────────────────────────────────────────────────┘

基于 Isaac Lab Arena 0.1.0 三个核心模块:
  ╔══════════╦══════════════════════════════════════════╗
  ║ Scene    ║ galileo_locomanip + brown_box + bin    ║
  ║ Embodiment ║ g1_wbc_pink (宇树 G1 + WBC全身控制)   ║
  ║ Task     ║ G1LocomanipPickAndPlaceTask             ║
  ╚══════════╩══════════════════════════════════════════╝

运行方法
════════

【环境要求】
  - Conda 环境: env_isaaclab (Python 3.11 + Isaac Lab 0.47.2)
  - GPU: NVIDIA RTX 4090 (23GB VRAM)，需要 ≥8GB 空闲
  - 显示器: 需要 X11 显示 (DISPLAY 环境变量已设置，否则使用 --headless)

【步骤 1: 激活环境】
  $ conda activate env_isaaclab
  $ cd /home/rq/文档/project/arena_0_1_0

【步骤 2: GUI 模式 (Mock 后端)】
  $ python arena_g1_locomanip_pnp.py --no_headless --backend mock --max_steps 500

  预期效果:
  - 弹出 Isaac Sim 渲染窗口
  - 显示 galileo_locomanip 操作场景（开放式空间，含操作台）
  - 宇树 G1 人形机器人站在场景中心 (0, 0.18, 0)
  - 棕色箱子放在操作台上 (0.5785, 0.18, 0.0707)
  - 蓝色分拣箱放在地面 (-0.245, -1.627, -0.264)
  - G1 先导航到操作台，转向，抓取箱子，再导航到分拣箱放置
  - 任务最长 20 秒（episode_length_s=20.0）

【步骤 3: Headless 模式 (后台运行)】
  $ python arena_g1_locomanip_pnp.py --backend mock --max_steps 500

【程序启动流程 (7 步时序)】
  第 1 步  AppLauncher         ← 启动 Isaac Sim (Kit + PhysX + 渲染)
  第 2 步  AssetRegistry       ← 注册资产
           get_asset_by_name("galileo_locomanip")()  → 操作场景
           get_asset_by_name("brown_box")()           → 棕色箱子
           get_asset_by_name("blue_sorting_bin")()    → 蓝色分拣箱
           get_asset_by_name("g1_wbc_pink")()          → 宇树 G1
  第 3 步  Scene 组合           ← Scene(assets=[...])
  第 4 步  G1LocomanipPickAndPlaceTask ← 定义移动操作抓取放置任务
  第 5 步  ArenaEnvBuilder      ← orchestrate() + compose_manager_cfg()
  第 6 步  ManagerBasedRLEnv    ← 创建仿真环境
  第 7 步  ControlLoop          ← while loop: observe → infer → decode → execute

【动作空间】
  G1 全身控制 (WBC + PINK IK):
  - 下肢行走: 腿部关节 × 6×2 = 12 维
  - 双臂 IK: 上肢关节 × 7×2 = 14 维 (PINK IK 控制)
  - 手部关节: 手指关节若干维
  - 总计: ~50+ 维 (根据 G1 WBC 配置动态确定)

【成功条件】
  objects_in_proximity:
  - 棕色箱子进入蓝色分拣箱的 X 方向 ±26cm, Y 方向 ±13cm, Z 方向 ±15cm 范围内
  - 判定: "箱子已放入分拣箱" → success=True

【失败条件】
  1. 箱子掉落 (root_height_below_minimum): 箱子中心 Z < -0.6m
     → object_dropped=True
  2. 超时: episode_length_s=20.0s 到期

本脚本在整体架构中的位置
════════════════════════
  本脚本是【独立仿真实验】入口：一次性跑通 Arena 三核心 + Isaac Sim，
  用于验证"物理场景 / 资产加载 / WBC+IK 控制链 / PnP 任务终止条件"。
  它不引入 LeRobot，也不对外暴露 Gymnasium 接口。

  同目录下的 arena_g1_locomanip_pnp_lerobot.py 是同一套逻辑的
  LeRobot/Gymnasium 兼容封装，两者关系如下：

    ┌────────────────────┬──────────────────────────┬────────────────────────────┐
    │ 维度               │ 本文件 (pnp.py)           │ pnp_lerobot.py              │
    ├────────────────────┼──────────────────────────┼────────────────────────────┤
    │ 组织形态           │ main() 脚本 + argparse    │ G1LocomanipPnPLeRobotEnv 类 │
    │ 闭环驱动           │ arena.client.ControlLoop  │ 调用方自己写 for 循环        │
    │ 接口契约           │ 自定义 G1LocomanipRobot   │ Gymnasium reset/step        │
    │ 典型用途           │ 人工调试 / 单次实验       │ lerobot-eval / VLA 训练评估 │
    │ success 快速验证   │ --success_test            │ --success_test / run_success_test() │
    └────────────────────┴──────────────────────────┴────────────────────────────┘

  两者的场景坐标、接近度阈值、四层 monkey-patch 完全一致；
  修改任何一处（例如阈值或初始位姿）都必须同步另一处，否则实验不可比。

依赖哪些外部系统
════════════════
  - NVIDIA Isaac Sim 5.1 + Isaac Lab 0.47.2：AppLauncher(Kit/PhysX)、
    ManagerBasedRLEnv、isaaclab.utils.assets.retrieve_file_path。
  - isaaclab_arena 0.1.0：AssetRegistry / Scene / IsaacLabArenaEnvironment /
    ArenaEnvBuilder / G1LocomanipPickAndPlaceTask。
  - isaaclab_arena_g1：G1 全身控制器（下肢 WBC 策略 + 上肢 PINK IK）。
  - 第三方库：pink(IK)、qpsolvers、scipy、numpy、torch；HTTP 后端还需 json_numpy。
  - 本仓库运行时包：arena.adapter / arena.client / arena.config / arena.backends /
    arena.types。
  - 本仓库辅助模块：_g1_wbc_stub.py（下肢策略占位实现）。
  - 本地资产副本：/tmp/unitree_ros_g1/robots/g1_description/（由
    download_g1_wbc_assets.py 从 Unitree GitHub 克隆）。

运行前提
════════
  1. 已激活 conda 环境 env_isaaclab，且 `arena` 运行时包在 PYTHONPATH 上；
  2. 已执行 download_g1_wbc_assets.py，使
     /tmp/unitree_ros_g1/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf 存在；
     若缺失，patch 仍会安装，但 URDF 加载会失败（日志会打印 "缺失!"）；
  3. 有可用 NVIDIA GPU（≥8GB 空闲显存）；--no_headless 时还需要 X11 DISPLAY；
  4. 三层 patch 必须全部生效：资产重定向、WBC stub、PINK IK 求解器替换。

已知边界（诚实声明）
══════════════════
  - 下肢由 _g1_wbc_stub.py 的固定站立姿态驱动，机器人【不会行走】，
    导航子目标不会被真正执行；因此 mock 闭环下 success 恒为 false 属预期，
    它只证明"仿真与接口链路可跑通"，不代表任务已实现。
  - 因 Isaac Sim 5.1 的 camera data 0 dim 问题，相机被强制关闭
    （enable_cameras=False），观测中的图像是占位张量，不能喂给真实视觉策略。

══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse  # 命令行参数解析
import json      # 结果序列化
import logging   # 日志输出
from typing import Any, Dict, Optional

import numpy as np   # 数组运算
import torch          # GPU 张量操作

# 配置日志格式: 时间 [级别] 消息
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# G1 移动操作机器人包装器 — 适配 ARENA RobotInterface
# ═══════════════════════════════════════════════════════════════════

class G1LocomanipRobot:
    """
    宇树 G1 人形机器人包装器（移动操作模式）。

    核心职责:
      - 将 CUDA 张量转换为 numpy 数组 (ARENA 框架全部使用 numpy)
      - 将 Isaac Lab 嵌套观测格式映射为 ARENA 规范格式
        {images: {"head": ndarray, "wrist": ndarray}, state: ndarray}
      - 动作维度补齐到 G1 WBC 动作空间的完整维度
      - 检测 episode 终止 (success/object_dropped/timeout)

    G1 WBC Pink 控制器特点:
      - 全身控制 (Whole Body Control): 同时控制下肢行走 + 上肢操作
      - PINK IK: 双臂末端执行器 (EEF) 的逆运动学
      - 导航: 通过 P-Controller 实现步行轨迹跟踪
    """

    # ── 观测键名映射 ──────────────────────────────────────────────
    # 设计意图: Isaac Lab 环境在不同版本/不同 embodiment 下会给出不同的观测键名,
    # 因此这里用"候选键名元组 + 首次命中"的策略做兼容, 而不是硬绑定某一个键。
    # 元组内部按优先级从高到低排列, get_observation() 命中即 break。
    FULL_IMAGE_KEYS = (          # 第三人称视图 (头戴相机)
        "agentview_image", "robot_pov_cam_rgb", "full_image", "image", "rgb", "head_image"
    )
    WRIST_IMAGE_KEYS = (         # 腕部相机 (眼在手上)
        "wrist_image", "robot0_eye_in_hand_image", "wrist"
    )
    STATE_KEYS = (               # 本体感知状态
        "joint_pos", "state", "robot_state", "proprio", "policy"
    )

    def __init__(self, env, instruction: str = "", max_steps: int = 500):
        """
        初始化 G1 移动操作机器人包装器。

        参数:
            env: Isaac Lab ManagerBasedRLEnv 实例
            instruction: 自然语言任务描述
            max_steps: 最大步数 (Loco-Manipulation 任务建议 ≥500 步)
        """
        self.env = env
        self.instruction = instruction
        self.max_steps = max_steps
        self.steps = 0
        # 观测缓存: reset()/execute() 每次都会刷新, get_observation() 只读该缓存。
        # 这样 ARENA 的"observe → infer → act"两段式调用不会重复 step 环境。
        self._last_obs: Optional[Dict[str, Any]] = None
        self._action_dim = self._get_action_dim()

    def _get_action_dim(self) -> int:
        """获取 G1 WBC 动作空间维度 (~50+ 维，含下肢+上肢+手部)"""
        try:
            return int(self.env.action_space.shape[-1])
        except Exception:
            # 回退原因: 部分 Arena 版本在 init 阶段 action_space 仍为 None 或
            # 形状未定; 此时用 G1 的典型维度兜底, 保证 execute() 不因取维度而崩。
            return 50  # G1 默认回退值

    # ── RobotInterface 接口实现 ───────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """重置环境并返回初始观测。"""
        result = self.env.reset()
        self._last_obs = self._unwrap(result)
        self.steps = 0
        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """
        从缓存的原始观测提取 ARENA 规范格式。

        返回:
            {"images": {"head": ndarray, ...}, "state": ndarray}

        注意:
          Loco-Manipulation 任务的 state 包含更丰富的本体感知信息:
          - 全身关节位置 (~50+ 维)
          - 机器人世界位姿 (位置 + 旋转)
          - 左右末端执行器位姿
          - 导航状态 (目标位置、朝向)

        失败行为:
          - 未调用 reset() 时抛 RuntimeError（fail-fast，避免下游拿到空观测）;
          - 图像/状态键全部缺失时不抛异常，而是退化为占位图像与空 state，
            让 mock/无相机场景仍能跑通（代价是策略拿不到有效感知）。
        """
        if self._last_obs is None:
            raise RuntimeError("请先调用 reset() 再调用 get_observation()")

        images: Dict[str, np.ndarray] = {}
        # 提取头戴相机图像
        for key in self.FULL_IMAGE_KEYS:
            val = self._last_obs.get(key)
            if val is not None:
                images["head"] = self._to_numpy(val)
                break
        # 提取腕部相机图像
        for key in self.WRIST_IMAGE_KEYS:
            val = self._last_obs.get(key)
            if val is not None:
                images["wrist"] = self._to_numpy(val)
                break
        # 兜底: 一张全黑 224x224x3 uint8 图。原因是 camera 被强制关闭后
        # 观测里没有任何图像键，若不补占位，ControlLoop 的编码器会 KeyError。
        if not images:
            images["head"] = np.zeros((224, 224, 3), dtype=np.uint8)

        # 提取本体感知状态
        # G1 WBC 环境返回嵌套观测: {"policy": {"robot_joint_pos": tensor, ...}}
        # 先按"扁平键"取，再退到 policy 子字典，兼容 Arena 的两种观测组织方式。
        state = self._last_obs.get("robot_joint_pos")
        if state is None:
            policy = self._last_obs.get("policy")
            if isinstance(policy, dict):
                state = policy.get("robot_joint_pos")
        if state is not None:
            state = self._to_numpy(state)
        else:
            # 空 state: 让调用方至少能拿到形状合法的 (0,) 向量而不是 None。
            state = np.zeros(0, dtype=np.float32)

        return {"images": images, "state": np.asarray(state, dtype=np.float32).reshape(-1)}

    def execute(self, action: np.ndarray) -> None:
        """
        执行单个动作（G1 全身控制动作）。

        参数:
            action: 动作向量 (~50+ 维)
                    = 下肢行走 (12维) + 双臂IK (14维) + 手部关节 + 导航参数

        处理流程:
          1. 补齐/截断到 action_dim
          2. 转 GPU 张量 → env.step()
          3. 更新观测缓存 → 步数 +1

        失败行为: env.step() 抛出的异常不在此处捕获，直接向上传播，
                  由 main() 的 try/except 统一记录并进入清理流程。
        """
        # ── 维度对齐 ──────────────────────────────────────────────
        # 策略侧维度与 SDK 动作契约可能不一致(例如 VLA 只输出 23 维子集),
        # 因此这里"短的右侧补 0、长的直接截断"，保证送进 env.step 的宽度恒定。
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if len(arr) < self._action_dim:
            arr = np.pad(arr, (0, self._action_dim - len(arr)))
        elif len(arr) > self._action_dim:
            arr = arr[: self._action_dim]

        # 关节限位裁剪
        # 目的: mock 随机动作或 VLA 越界输出会直接触发 PhysX/IK 的非法输入，
        # 裁剪到 action_space 的 [low, high] 可把错误限制在"动作被削顶"而非崩溃。
        if hasattr(self.env, "action_space") and self.env.action_space is not None:
            low = np.asarray(self.env.action_space.low).reshape(-1)
            high = np.asarray(self.env.action_space.high).reshape(-1)
            m = min(len(arr), len(low))
            arr[:m] = np.clip(arr[:m], low[:m], high[:m])

        # ManagerBasedRLEnv 期望 (num_envs=1, action_dim) 的批张量，因此 unsqueeze(0)。
        action_tensor = torch.from_numpy(arr).float().unsqueeze(0)
        if hasattr(self.env.unwrapped, "device"):
            action_tensor = action_tensor.to(self.env.unwrapped.device)

        # inference_mode: 纯推理，禁用 autograd，省显存并避免被动建图。
        with torch.inference_mode():
            result = self.env.step(action_tensor)

        self._last_obs = self._unwrap(result)
        self.steps += 1

    def is_running(self) -> bool:
        """判断 episode 是否应继续。"""
        return self.steps < self.max_steps and not self.task_finished()

    def task_finished(self) -> bool:
        """
        判断任务是否已完成。

        成功条件: 箱子在分拣箱接近范围内
          → terminated["success"] = True

        失败条件: 箱子掉落 (Z < -0.6m)
          → terminated["object_dropped"] = True

        超时条件: episode 时长 > 20.0s
          → truncated = True

        实现说明: 这里用"任一终止键为真即结束"的宽松判据，而不是精确区分
        success 与 failed。原因见下方注释——调用方(ControlLoop)只关心是否结束。
        """
        if self._last_obs is None:
            return False
        # 注意: success / object_dropped / timeout 三类信号都会让本函数返回 True;
        # 真正的成败细分由 Arena 的 TerminationManager 写进 info，本包装器不透传。
        for key in ("terminated", "truncated", "done", "success"):
            v = self._last_obs.get(key)
            if v is not None and bool(np.asarray(v).any()):
                return True
        return False

    # ── 内部工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _to_numpy(val):
        """把 torch.Tensor / list / 标量统一转成 numpy 数组。

        参数:
            val: 任意观测值（可能是 CUDA 张量、numpy 数组或 Python 序列）。

        返回:
            numpy 数组；转换失败时退化为 float32 数组。

        说明: 张量先 .cpu() 再 .numpy()，否则会抛
              "can't convert cuda:0 device type tensor to numpy"。
        """
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, (torch.Tensor,)):
            return val.cpu().numpy()
        try:
            return np.asarray(val)
        except Exception:
            return np.asarray(val, dtype=np.float32)

    @staticmethod
    def _unwrap(result) -> Dict[str, Any]:
        """统一 Isaac Lab 的多种 step/reset 返回格式为扁平字典。

        参数:
            result: env.reset()/env.step() 的原始返回值，可能是
                    2/4/5 元组、纯 dict，或其它可忽略对象。

        返回:
            扁平字典：观测键 + terminated/truncated/done/reward 等元信息
            （元信息已转 numpy，便于后续 bool()/.any() 判断）。

        容错策略: 元组元素个数决定解析分支；无法识别时返回空字典而不是报错，
                  这样不同 Isaac Lab 版本都能跑，代价是可能静默丢掉信息。
        """
        if isinstance(result, tuple):
            # Gymnasium 5 元组: (obs, reward, terminated, truncated, info)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                merged = dict(obs) if isinstance(obs, dict) else {}
                merged["terminated"] = G1LocomanipRobot._to_numpy(terminated)
                merged["truncated"] = G1LocomanipRobot._to_numpy(truncated)
                merged["reward"] = G1LocomanipRobot._to_numpy(reward)
                return merged
            # 旧版 4 元组: (obs, reward, done, info)
            if len(result) == 4:
                obs, reward, done, info = result
                merged = dict(obs) if isinstance(obs, dict) else {}
                merged["done"] = G1LocomanipRobot._to_numpy(done)
                merged["reward"] = G1LocomanipRobot._to_numpy(reward)
                return merged
            # 其余情况: 把第 0 位当观测、第 1 位当 info，尽量榨出可用信息。
            obs, info = result[0], result[1] if len(result) > 1 else {}
            merged = dict(obs) if isinstance(obs, dict) else {}
            if isinstance(info, dict):
                for k, v in info.items():
                    merged[k] = G1LocomanipRobot._to_numpy(v) if isinstance(v, (torch.Tensor,)) else v
            return merged
        if isinstance(result, dict):
            return result
        return {}


# ═══════════════════════════════════════════════════════════════════
# Monkey-Patch 栈 — 修复"在线资源不可达 + 依赖版本漂移"两类缺陷
#   ① 资产重定向: Nucleus URL → 本地 Unitree GitHub 克隆
#   ② WBC 下肢策略: G1HomiePolicyV2(ONNX) → _g1_wbc_stub 固定站立
#   ③ PINK IK 求解器: solver="osqp" → "quadprog"
# ═══════════════════════════════════════════════════════════════════
# 主程序 — 基于 Arena SDK 三核心模块
#   Scene: galileo_locomanip + brown_box + blue_sorting_bin
#   Embodiment: g1_wbc_pink (宇树 G1 + 全身控制 + PINK IK)
#   Task: G1LocomanipPickAndPlaceTask (箱子接近度检测)
# ═══════════════════════════════════════════════════════════════════

def _patch_g1_wbc_assets():
    """
    Monkey-patch: 将 G1 WBC 的 Nucleus 资产 URL 重定向到本地 GitHub 下载的 URDF。

    原因: omniverse://isaac-dev.ov.nvidia.com 是 NVIDIA 内部开发服务器，
          当前环境无法访问。使用官方 Unitree GitHub 仓库中的 URDF 替代。

    URDF 来源: https://github.com/unitreerobotics/unitree_ros
    本地路径: /tmp/unitree_ros_g1/robots/g1_description/

    为什么重定向到"GitHub 克隆"而不是别处:
      Arena SDK 里写死的 Nucleus 路径同时承载两类东西——机器人模型描述
      (URDF) 与配套资产 (meshes/robot_model/wbc_policy 等)。离线环境里
      Unitree 官方仓库是唯一能拿到同一台 G1 的完整 URDF+mesh 的公开来源，
      因此用 download_g1_wbc_assets.py 预先克隆到 /tmp，再在运行时把
      URL 前缀命中者改为本地绝对路径。

    如何区分 URDF 文件与资产目录:
      只看 basename 里是否含 "urdf" 子串——
        - 命中  → 一定是单个模型描述文件，返回 _g1_urdf 这个具体文件路径；
        - 未命中 → 视作目录型资产（meshdir / robot_model / wbc_policy），
                   返回 _g1_pkg_dir 包目录。
      之所以能这样"粗判"，是因为 meshes 用相对 meshdir="meshes" 解析，
      只要 URDF 与 meshes/ 同处一个包目录即可被正确找到。

    补丁时机: 必须在 AppLauncher 启动之后（carb/Kit 就绪）、资产加载之前调用，
              本函数在 main() 中正是插在这两个时刻之间。

    失败行为: 若本地 URDF 不存在，本函数不抛异常，只把日志标记为 "缺失!"；
              真正的失败会延后到资产加载阶段以 URDF 解析错误的形式暴露。
    """
    import os as _os

    # 本地 URDF 与包目录（包目录是 meshdir="meshes" 的父目录）
    _g1_urdf = "/tmp/unitree_ros_g1/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf"
    _g1_pkg_dir = "/tmp/unitree_ros_g1/robots/g1_description"  # meshdir="meshes" 的父目录

    import isaaclab.utils.assets as _assets_mod

    # 先保存原函数：命中前缀之外的一切路径仍需走原始逻辑（例如别的基础资产）。
    _original_retrieve = _assets_mod.retrieve_file_path
    _NUCLEUS_G1_PREFIX = "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets"

    def _patched_retrieve_file_path(path: str, download_dir=None, force_download=True) -> str:
        """代理 retrieve_file_path：G1 资产走本地副本，其余路径原样透传。

        参数:
            path: 原始资产路径（可能是 omniverse:// URL 或普通路径）。
            download_dir: 透传给原始实现的下载目录。
            force_download: 透传给原始实现的强制下载标志。

        返回:
            本地绝对路径（命中 Nucleus 前缀时）或原始实现的返回值。
        """
        if path.startswith(_NUCLEUS_G1_PREFIX):
            if "urdf" in _os.path.basename(path).lower():
                logging.getLogger("arena").info("  → 使用本地 G1 URDF: %s", _g1_urdf)
                return _os.path.abspath(_g1_urdf)
            # 目录路径 (robot_model, wbc_policy 等) → 返回包目录
            logging.getLogger("arena").info("  → 使用本地 G1 包目录: %s", _g1_pkg_dir)
            return _os.path.abspath(_g1_pkg_dir)
        return _original_retrieve(path, download_dir=download_dir, force_download=force_download)

    # 替换模块属性。注意: 只对"通过模块属性查找"的调用者生效；
    # 若某模块已用 `from ... import retrieve_file_path` 绑定了别名，则不受影响。
    _assets_mod.retrieve_file_path = _patched_retrieve_file_path
    _ok = _os.path.isfile(_g1_urdf)
    logging.getLogger("arena").info(
        "✓ G1 WBC 资产 patch 已应用 (URDF: %s)", "OK" if _ok else "缺失!"
    )

def main():
    """
    Arena 0.1.0 G1 移动操作抓取放置仿真实验入口。

    执行流程 (7 步):
      ① 启动 Isaac Sim          → AppLauncher
      ② 构建场景 (Scene)        → galileo_locomanip + brown_box + bin
      ③ 构建机器人 (Embodiment) → g1_wbc_pink (宇树 G1 + WBC)
      ④ 定义任务 (Task)          → G1LocomanipPickAndPlaceTask
      ⑤ 生成配置                 → ArenaEnvBuilder → ManagerBasedRLEnvCfg
      ⑥ 创建仿真环境             → ManagerBasedRLEnv(cfg=cfg)
      ⑦ 闭环执行                 → ControlLoop.run()

    参数: 无（全部来自 argparse 命令行）。

    返回: 无返回值；实验结果通过日志与 json.dumps 打印。

    失败行为: 主体逻辑包在 try/except 中——异常被 logger.exception 记录后
              吞掉，finally 负责 env.close() 与 sim_app.close()，
              因此进程以退出码 0 结束（排障时需看日志而非退出码）。

    两种运行模式:
      - 默认: ControlLoop 闭环（mock 随机动作或 http VLA 服务）;
      - --success_test: 跳过策略，直接把箱子瞬移到分拣箱附近，
        只验证 PnP 任务的 success 终止条件是否可被触发（见文末分节注释）。
    """

    # ── 命令行参数 ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="ARENA 0.1.0 — G1 移动操作箱子抓取与放置仿真实验"
    )
    parser.add_argument(
        "--backend", default="mock", choices=["mock", "http"],
        help='策略后端: "mock"=随机动作 / "http"=VLA服务器',
    )
    parser.add_argument(
        "--server_url", default="http://127.0.0.1:8777/act",
        help="VLA 策略服务器地址",
    )
    parser.add_argument(
        "--instruction", default="pick up the brown box and place it in the blue bin",
        help="自然语言任务指令",
    )
    parser.add_argument(
        "--max_steps", type=int, default=500,
        help="最大仿真步数 (Loco-Manipulation 需 ≥500)",
    )
    # 注意: store_true + default=True 的组合意味着 --headless 是"空操作"，
    # 真正有用的是反向开关 --no_headless。此处保持原样以避免改变 CLI 语义。
    parser.add_argument("--headless", action="store_true", default=True,
                        help="无头模式")
    parser.add_argument("--no_headless", dest="headless", action="store_false",
                        help="GUI 模式")
    parser.add_argument(
        "--object", type=str, default="brown_box",
        help="要抓取的物体 (默认: brown_box)",
    )
    parser.add_argument(
        "--embodiment", type=str, default="g1_wbc_pink",
        help="机器人实施例 (默认: g1_wbc_pink = 宇树G1 + WBC全身控制)",
    )
    parser.add_argument(
        "--success_test", action="store_true", default=False,
        help="★ 测试模式: 直接将箱子瞬移到分拣箱范围内，验证 PnP 任务 success 条件",
    )
    args = parser.parse_args()

    # ══════════════════════════════════════════════════════════════
    # 第 1 步: 启动 Isaac Sim 仿真引擎
    # ══════════════════════════════════════════════════════════════
    from isaaclab.app import AppLauncher

    # 完全禁用相机 (Isaac Sim 5.1 camera data 0 dim workaround)
    # 无论 headless 还是 GUI，相机都会触发 syntheticdata 崩溃
    enable_cameras = False
    logger.info(
        "▶ 启动 Isaac Sim (headless=%s, cameras=%s)...",
        args.headless, enable_cameras,
    )
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=enable_cameras)
    sim_app = app_launcher.app

    # ── Monkey-Patch ①: 资产重定向 ────────────────────────────────
    # Monkey-patch: 在 Isaac Sim 启动后（carb 可用）、WBC 资产加载前，重定向 Nucleus URL
    _patch_g1_wbc_assets()

    # ── Monkey-Patch ②: WBC 下肢策略 Stub ─────────────────────────
    # _g1_wbc_stub.apply_patch() 把 Arena 工厂里的 G1HomiePolicyV2 换成
    # "固定站立姿态"的本地实现，因为官方 ONNX 权重 (stand.onnx/walk.onnx)
    # 位于不可达的 Nucleus 上。
    # 边界: Stub 只输出恒定站立动作，机器人【不会行走】——它保证环境能建起来、
    #       场景/任务链路可验证，但 navigate_cmd 不会被真正执行。
    from _g1_wbc_stub import apply_patch as _apply_wbc_stub
    _apply_wbc_stub()

    # ── Monkey-Patch ③: PINK IK 求解器替换 ────────────────────────
    # monkey-patch: qpsolvers 4.x 中 osqp 不可用，替换为 quadprog
    #
    # 背景: qpsolvers 从 1.x 升级到 4.x 后不再把 osqp 作为可选后端随包安装/导出，
    #       而 isaaclab_arena_g1 的上肢控制器里硬编码了 solver="osqp"，
    #       于是 solve_ik 会在解析求解器名时直接抛错（"solver osqp not found"）。
    #       quadprog 是同一问题的等价 QP 后端，且在 qpsolvers 4.x 中仍可用，
    #       因此运行时把 solver 名改写为 quadprog。
    import importlib as _il
    _pink_ik = _il.import_module("pink.solve_ik")
    _pink_ik_orig = _pink_ik.solve_ik
    def _patched_solve_ik(*args, **kwargs):
        """包装 pink.solve_ik：把所有 osqp 求解请求改写成 quadprog。

        参数: 与 pink.solve_ik 完全一致（*args/**kwargs 透传）。
        返回: pink.solve_ik 的原始返回值（关节速度/位形解）。
        """
        kwargs.setdefault("solver", "quadprog")
        if kwargs.get("solver") == "osqp":
            kwargs["solver"] = "quadprog"
        return _pink_ik_orig(*args, **kwargs)
    _pink_ik.solve_ik = _patched_solve_ik

    # 同时 patch 导入的别名 (from pink.solve_ik import solve_ik 的引用)
    # 为什么需要"运行时"这一层: 只改源文件对"已经导入并绑定过别名"的调用者无效，
    # 而 Arena 的控制器在模块顶层就做了 from ... import，故必须在运行时
    # 覆盖 solve_ik 模块属性及其可能的顶层别名，形成双保险（源文件 patch 由
    # pnp_lerobot.py 的 _patch_source_files() 负责；本文件没有做静态改写）。
    import pink as _pink_mod
    def _safe_replace(mod, name, fn):
        """把 mod.name 替换为转发到 fn 的包装函数。

        参数:
            mod: 目标模块对象。
            name: 要替换的属性名。
            fn: 实际执行的函数（这里是被 patch 过的 solve_ik）。

        失败行为: 模块不存在该属性时静默忽略（Arena 版本差异下可能没有顶层别名）。
        """
        try:
            orig = getattr(mod, name)
            setattr(mod, name, lambda *a, **kw: fn(*a, **kw))
        except Exception:
            pass
    _safe_replace(_pink_mod, "solve_ik", _patched_solve_ik)
    logger.info("✓ PINK IK 求解器已切换为 quadprog (osqp 不可用)")

    # ══════════════════════════════════════════════════════════════
    # 第 2 步: Scene 模块 — 构建操作场景
    # ══════════════════════════════════════════════════════════════
    # 场景组成:
    #   - galileo_locomanip: 开放式操作空间背景 (地面 + 操作台 + 围栏)
    #   - brown_box: 棕色箱子 (要抓取的物体)
    #   - blue_sorting_bin: 蓝色分拣箱 (目标放置位置)
    import argparse as _argparse
    from isaaclab_arena.assets.asset_registry import AssetRegistry

    # AssetRegistry 是 SDK 的资产工厂：按名字查表并实例化，避免在脚本里
    # 硬编码 USD/URDF 路径，从而与 Arena 的资产版本解耦。
    asset_registry = AssetRegistry()

    # ── 背景: Galileo Loco-Manipulation 场景 ───────────────────────
    # 该场景包含:
    #   - 大型操作台面 (高度 ~0.9m)
    #   - 开放式空间 (足够 G1 在操作台和分拣箱之间导航)
    #   - 围栏/标记 (辅助定位)
    logger.info("▶ 加载 Galileo Loco-Manipulation 操作场景...")
    background = asset_registry.get_asset_by_name("galileo_locomanip")()

    # ── 抓取物体: 棕色箱子 ────────────────────────────────────────
    # brown_box 的位置:
    #   - X=0.5785m (操作台前方偏右)
    #   - Y=0.18m (操作台中央)
    #   - Z=0.0707m (操作台面上方 7cm)
    #   - 无旋转 (平放)
    logger.info("▶ 加载棕色箱子 (brown_box)...")
    pick_up_object = asset_registry.get_asset_by_name(args.object)()
    # Pose 采用 rotation_wxyz 命名，即 (w,x,y,z)。这里 (0,0,1,0) 表示绕 Y 轴 180°，
    # 使箱子朝向 G1（与 blueprint 中的默认摆放一致）。
    pick_up_object.set_initial_pose(
        _import_pose()(
            position_xyz=(0.5785, 0.18, 0.0707),
            rotation_wxyz=(0.0, 0.0, 1.0, 0.0),
        )
    )

    # ── 目标物体: 蓝色分拣箱 ──────────────────────────────────────
    # blue_sorting_bin 的位置:
    #   - X=-0.245m (G1 后方偏左)
    #   - Y=-1.627m (操作台左侧 1.6m)
    #   - Z=-0.264m (地面高度)
    #   - 无旋转
    logger.info("▶ 加载蓝色分拣箱 (blue_sorting_bin)...")
    blue_sorting_bin = asset_registry.get_asset_by_name("blue_sorting_bin")()
    # 单位四元数 (w=1) → 不旋转。该坐标与 success_test 中的 _bin_init 硬编码值一致，
    # 改一处必须同步改另一处。
    blue_sorting_bin.set_initial_pose(
        _import_pose()(
            position_xyz=(-0.2450, -1.6272, -0.2641),
            rotation_wxyz=(0.0, 0.0, 0.0, 1.0),
        )
    )

    # ── 组合场景 ──────────────────────────────────────────────────
    from isaaclab_arena.scene.scene import Scene

    # Scene 只做"资产集合"的声明，不做实例化；真正的 USD 引用与
    # 环境数量在 ArenaEnvBuilder.compose_manager_cfg() 阶段才落到 cfg 上。
    assets = [background, pick_up_object, blue_sorting_bin]
    scene = Scene(assets=assets)

    # ══════════════════════════════════════════════════════════════
    # 第 3 步: Embodiment 模块 — 宇树 G1 + WBC 全身控制
    # ══════════════════════════════════════════════════════════════
    # g1_wbc_pink = 宇树 G1 人形机器人 + 全身控制 (Whole Body Control)
    #   - 下肢: 6 DOF × 2 腿部 = 12 维行走
    #   - 上肢: 7 DOF × 2 臂部 = 14 维 (PINK IK 控制)
    #   - 手部: 手指关节若干维
    #   - 导航: WBC 使用 P-Controller 跟踪步行子目标
    logger.info("▶ 加载宇树 G1 WBC Pink 实施例 (全身控制 + PINK IK)...")
    # enable_cameras 透传给 embodiment：相机资产在关闭相机时不应被创建，
    # 否则 syntheticdata 仍会初始化并触发 0 维崩溃。
    embodiment = asset_registry.get_asset_by_name(args.embodiment)(
        enable_cameras=enable_cameras
    )
    # 设置 G1 的初始位置 (场景中心)
    # 单位四元数 w=1 → G1 朝向 +X，正对操作台（箱子在 X 正向）。
    embodiment.set_initial_pose(
        _import_pose()(
            position_xyz=(0.0, 0.18, 0.0),
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    )

    # ══════════════════════════════════════════════════════════════
    # 第 4 步: Task 模块 — 移动操作箱子抓取与放置
    # ══════════════════════════════════════════════════════════════
    # G1LocomanipPickAndPlaceTask 是内置的移动操作抓取放置任务:
    #
    # 成功条件 (objects_in_proximity):
    #   - 箱子中心 X 在分拣箱 ±26cm 范围内 → max_x_separation=0.260
    #   - 箱子中心 Y 在分拣箱 ±13cm 范围内 → max_y_separation=0.130
    #   - 箱子中心 Z 在分拣箱 ±15cm 范围内 → max_z_separation=0.150
    #   - 三轴同时满足 → success=True
    #
    # 失败条件 (object_dropped):
    #   - 箱子掉落到地面以下 (Z < -0.6m) → object_dropped
    #
    # episode 时长: 20.0 秒 (含导航+抓取+放置全过程)
    from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import (
        G1LocomanipPickAndPlaceTask,
    )

    # 阈值与超时由 SDK 任务的默认值提供，脚本只显式覆盖 episode_length_s：
    # 20s 是"导航→转向→抓取→搬运→放置"全流程的时间上限。
    task = G1LocomanipPickAndPlaceTask(
        pick_up_object,       # 要抓取的箱子
        blue_sorting_bin,     # 目标分拣箱
        background,           # 操作场景
        episode_length_s=20.0,# 20 秒超时
    )
    logger.info(
        "▶ 任务: G1LocomanipPickAndPlaceTask "
        "(接近度阈值: X±26cm, Y±13cm, Z±15cm, 超时=20s)"
    )

    # ══════════════════════════════════════════════════════════════
    # 第 5 步: 构建 Isaac Lab 仿真配置
    # ══════════════════════════════════════════════════════════════
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    # 组合场景 + 实施例 + 任务
    arena_env = IsaacLabArenaEnvironment(
        name="galileo_g1_locomanip_pick_and_place",
        embodiment=embodiment,
        scene=scene,
        task=task,
    )

    # ArenaEnvBuilder 需要一个"模拟 CLI 解析结果"的 Namespace（它内部按
    # 属性名读取参数，而不是按函数签名），因此这里手工构造。
    # 注意: enable_cameras=True 与上面的 enable_cameras=False 不一致——
    # 这是刻意保持的既有行为（潜在缺陷，见报告），未作修改。
    arena_argparse_ns = _argparse.Namespace(
        headless=args.headless,
        enable_cameras=True,
        device="cuda:0",
        num_envs=1,
        disable_fabric=True,
        seed=42,
        task=args.instruction,
        embodiment=args.embodiment,
        object=args.object,
        teleop_device=None,
        mimic=False,
        video=False,
        video_length=0,
        video_interval=0,
    )

    # orchestrate(): 按 Scene/Embodiment/Task 组装资产与 manager 项；
    # compose_manager_cfg(): 产出 ManagerBasedRLEnvCfg（纯配置对象，尚未建场景）。
    builder = ArenaEnvBuilder(arena_env, arena_argparse_ns)
    builder.orchestrate()
    cfg = builder.compose_manager_cfg()

    # ── 三项配置修复（SDK 与 Isaac Sim 5.1 的适配缺陷）─────────────
    # 修复 1: 移除 last_action 观测项 (0 维张量错误)
    #   Arena 默认把上一步动作放进 policy 观测。单环境 + 关闭相机后该张量会
    #   退化成 0 维，拼接观测时触发 shape 错误；本任务不需要 last_action，
    #   于是置 None 让 manager 跳过该观测项（None 而非删除，因为 ObservationGroupCfg
    #   的属性必须在位）。
    if hasattr(cfg.observations.policy, "actions"):
        cfg.observations.policy.actions = None
    # 修复 2: 禁用相机观测 (Isaac Sim 5.1 camera data 0 dim workaround)
    #   相机被整体关闭（enable_cameras=False），若仍保留 camera_obs 观测项，
    #   manager 会去读不存在的相机 buffer 而报 0 维错误；置 None 跳过。
    if hasattr(cfg.observations, "camera_obs"):
        cfg.observations.camera_obs = None
    # 修复 3: 单环境模式
    #   本实验是"单机器人单任务"调试场景，强制 num_envs=1，避免
    #   builder 默认的并行环境数导致显存翻倍与终止信号聚合歧义。
    cfg.scene.num_envs = 1

    # ══════════════════════════════════════════════════════════════
    # 第 6 步: 创建仿真环境
    # ══════════════════════════════════════════════════════════════
    env = ManagerBasedRLEnv(cfg=cfg)
    # 关闭 Isaac Sim 的"停止时自动关 app"回调: 由本脚本在 finally 中显式
    # 管理生命周期，否则 app 会在 env 仍被引用时提前销毁。
    env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

    # G1 移动操作动作空间 (~50+ 维)
    # = 下肢行走 (12维) + 双臂 IK (14维) + 手部关节 + 导航参数
    action_dim = int(env.action_space.shape[-1])
    logger.info(
        "✓ G1 移动操作环境已创建 — action_dim=%d (下肢+上肢+手部+导航)",
        action_dim,
    )

    # ══════════════════════════════════════════════════════════════
    # 第 7 步: ARENA 适配器 + 闭环执行
    # ══════════════════════════════════════════════════════════════
    from arena.adapter import EmbodimentAdapter
    from arena.client import ControlLoop, PolicyClient
    from arena.config import AdapterConfig, ClientConfig

    # adapter: 负责把 ARENA 规范观测编码成策略输入、把策略动作解码回
    #          SDK 动作契约（VLA 服务器与本地策略共用同一编码路径）。
    adapter = EmbodimentAdapter(
        AdapterConfig(robot_type="unitree_g1", action_dim=action_dim),
        instruction=args.instruction,
    )
    # robot: 上面定义的 RobotInterface 实现，封装 reset/observe/execute 语义。
    robot = G1LocomanipRobot(env, instruction=args.instruction, max_steps=args.max_steps)

    # ── 策略客户端 ────────────────────────────────────────────────
    # 两个后端共享同一 infer(obs, instruction) → ActionChunk 契约，
    # 因此 ControlLoop 无需感知后端差异。
    if args.backend == "mock":
        from arena.backends import build_backend
        from arena.config import ServerConfig
        from arena.types import ActionChunk

        # mock: 进程内随机动作后端，用于在没有 VLA 服务器时打通链路。
        backend = build_backend(ServerConfig(backend="mock"))
        # 把真实动作维度注入后端的随机采样器，保证生成的动作宽度=action_dim。
        backend.action_dim = action_dim

        # 进程内客户端：跳过 HTTP/序列化，直接调用 backend.infer。
        class InProcessClient:
            """进程内策略客户端：直接调用本地 mock 后端，不经 HTTP。

            与 PolicyClient 的差别: 没有网络与 JSON 编解码开销，但接口保持
            infer(obs, instruction) → ActionChunk 一致，因此 ControlLoop 无需感知。
            定义在 main() 内部是为了捕获闭包变量 backend / action_dim。
            """

            def infer(self, obs, instruction=None):
                """调用 mock 后端生成一批动作，并做四元数安全处理。

                参数:
                    obs: ARENA 观测对象（需支持 .to_dict()）。
                    instruction: 自然语言指令，透传给后端。

                返回:
                    ActionChunk(actions=[ndarray(action_dim), ...])。

                说明: 返回值先经 np.atleast_2d 统一成 (batch, dim)，
                      再逐条做四元数归一化（见下）。
                """
                _raw = backend.infer([obs.to_dict()], instruction or "")
                # backend.infer returns: numpy array (batch, dim) or list[ndarray]
                _actions = [np.asarray(a, dtype=np.float32) for a in np.atleast_2d(_raw)]
                # ★ 修复: Mock 随机动作中的四元数可能全为 0，
                # scipy.spatial.transform.Rotation.from_quat() 不允许零范数
                # action_constants.py: LEFT_QUAT=5:9, RIGHT_QUAT=12:16 (wxyz)
                #
                # 为什么偏偏是这两段是四元数:
                #   G1 WBC 动作契约按 23 维拼装——左臂 EEF 位置(2:5)+姿态(5:9)、
                #   右臂 EEF 位置(9:12)+姿态(12:16)。四元数只出现在姿态那 4 维，
                #   且约定为 (w,x,y,z)（与 Arena 的 rotation_wxyz 命名一致）。
                # 为什么零范数会报错:
                #   单位四元数空间是 S³，"全 0"不在其中；从四元数构造旋转时
                #   需要先除以范数，scipy 会显式抛
                #   "Found zero norm quaternions in `quat`"。
                #   mock 后端是均匀随机采样，四个分量同时取到 0 的概率极低但存在，
                #   一旦命中就整条链路崩溃，因此必须显式兜底。
                # 为什么补成 [1,0,0,0]:
                #   (w,x,y,z)=(1,0,0,0) 是 wxyz 约定下的"无旋转"（恒等旋转），
                #   是语义上最保守的合法值；非零范数时则归一化成合法单位四元数。
                for _a in _actions:
                    for _s in [(5, 9), (12, 16)]:
                        _q = _a[_s[0]:_s[1]]
                        _norm = np.linalg.norm(_q)
                        if _norm < 1e-6:
                            _a[_s[0]:_s[1]] = [1.0, 0.0, 0.0, 0.0]
                        else:
                            _a[_s[0]:_s[1]] = _q / _norm
                return ActionChunk(actions=_actions)
        client = InProcessClient()
    else:
        # http: 走真实 VLA 推理服务。json_numpy.patch() 让 numpy 数组
        # 可直接被 json 序列化，避免手动 base64/列化。
        import json_numpy
        json_numpy.patch()
        client = PolicyClient(ClientConfig(server_url=args.server_url))

    # ── 闭环执行 ──────────────────────────────────────────────────
    logger.info(
        "▶ 启动闭环控制 (max_steps=%d, backend=%s, episode_length=20s)...",
        args.max_steps, args.backend,
    )
    # ControlLoop 负责 observe → adapter.encode → client.infer → adapter.decode
    # → robot.execute 的循环，并调用 robot.is_running() 判断退出。
    loop = ControlLoop(
        robot, client, adapter,
        instruction=args.instruction,
        max_episodes=1,
    )

    try:
        if args.success_test:
            # ══════════════════════════════════════════════════════
            # ★ success_test 模式 ★
            # 目的: 绕开"策略是否会抓取"这个不确定因素，直接验证 PnP 任务的
            #       success 终止条件（objects_in_proximity）本身是否可用。
            # 做法: teleport 棕箱到分拣箱的接近度阈值内，步进若干帧让 PhysX
            #       传播位置，再分别用"直接读刚体位姿"与"完整 env.step 管线"
            #       两条路径判定 success。
            # ══════════════════════════════════════════════════════
            # 直接 teleport 棕箱到分拣箱的 X/Y/Z 接近度阈值内,
            # 步进 1 帧让 Termination Manager 检测到 objects_in_proximity,
            # 验证 PnP 任务成功终止条件可以正常触发
            logger.info("▶ [success_test] Reset 环境...")
            robot.reset()
            # 获取分拣箱和棕箱的 RigidObject
            # scene[name] 返回的是 Arena 注册的 RigidObject 句柄，
            # 可通过它 write_root_pose_to_sim() / 读 data.root_pos_w。
            _scene = env.unwrapped.scene
            _box_obj = _scene[blue_sorting_bin.name]   # 目标分拣箱
            _pick_obj = _scene[pick_up_object.name]     # 棕色箱子
            # 分拣箱初始世界坐标 (已知: blueprint 定义)
            # 直接复用设置初始位姿时的同一组常量，避免读场景带来的坐标系歧义。
            _bin_init = np.array([-0.2450, -1.6272, -0.2641], dtype=np.float32)
            # 将棕箱瞬移到分拣箱正上方 5cm (X/Y 0 偏差, Z+0.05 在 ±15cm 阈值内)
            # 实际偏移量为 0.10m（10cm）：既在 Z 阈值 0.150 之内，
            # 又留出余量，避免箱子落到分拣箱内壁后 Z 向超差。
            _target_pos = _bin_init + [0.0, 0.0, 0.10]
            logger.info("▶ [success_test] Teleport: 棕箱 → (%.3f, %.3f, %.3f)", *_target_pos)
            # write_root_pose_to_sim: (num_envs, 7) = [x,y,z, qx,qy,qz,qw]  ← Isaac Sim 用 xyzw 顺序!
            # 关键顺序问题: Arena 的 Pose 用 rotation_wxyz=(w,x,y,z)，而
            # PhysX/Isaac Sim 的 root pose 通道要求 (x,y,z,qx,qy,qz,qw)。
            # 两者相差一个首尾轮转；这里显式写作 xyzw，因此把 w 放在末尾、
            # 并对单位旋转填 (0,0,0,1)。
            _pose_7 = torch.tensor([[_target_pos[0], _target_pos[1], _target_pos[2],
                                      0.0, 0.0, 0.0, 1.0]],  # xyzw=(0,0,0,1)
                                     device=env.device, dtype=torch.float32)
            # 确保四元数范数=1
            # PhysX 要求单位四元数；显式归一化可在以后改常量时仍然安全。
            _quat = _pose_7[:, 3:]
            _pose_7[:, 3:] = _quat / torch.norm(_quat, dim=1, keepdim=True)
            _pick_obj.write_root_pose_to_sim(_pose_7)
            # simulator step: 让 PhysX 传播新位置
            # 为什么步进 3 帧而不是 1 帧: write_root_pose_to_sim 只把位姿写进
            # PhysX 的 staging buffer，需要 sim.step() 才会被积分/广播到
            # data.root_pos_w 等 fabric 视图；单帧有时仍读到旧值，
            # 连续 3 帧是经验上稳定可见的余量（且远小于 20s episode 预算）。
            for _ in range(3):
                robot.env.sim.step(render=False)
            # 直接读取 RigiObject 位置，手动计算接近度
            # 为什么要"手动算 ΔX/ΔY/ΔZ": 不走 TerminationManager 就不会产生
            # terminated 标志，而这一段的目的是独立验证 success 判据本身；
            # 用与 SDK 完全相同的阈值(0.260/0.130/0.150)复算一遍，
            # 可把"任务参数写错"与"终止管理器没触发"两类问题区分开。
            _p_pick = _pick_obj.data.root_pos_w.cpu().numpy().reshape(-1)
            _p_box = _box_obj.data.root_pos_w.cpu().numpy().reshape(-1)
            _dx = abs(_p_pick[0] - _p_box[0])
            _dy = abs(_p_pick[1] - _p_box[1])
            _dz = abs(_p_pick[2] - _p_box[2])
            _sep_ok = _dx < 0.260 and _dy < 0.130 and _dz < 0.150
            logger.info("✓ [success_test] 棕箱(%.3f, %.3f, %.3f) vs 分拣箱(%.3f, %.3f, %.3f)",
                        *_p_pick, *_p_box)
            logger.info("✓ [success_test] ΔX=%.3f (ok<0.260) ΔY=%.3f (ok<0.130) ΔZ=%.3f (ok<0.150) → %s",
                        _dx, _dy, _dz, "SUCCESS" if _sep_ok else "FALSE")
            _success = _sep_ok
            _truncated = False
            logger.info("✓ [success_test] Direct check: Terminated=%s", _success)

            # ★ 阶段 2: 通过 env.step() 触发完整管线 (WBC + PINK IK + Termination Manager)
            logger.info("▶ [success_test] 尝试 env.step() 走完整终止检测管线...")
            # 构建一个安全动作: navigate_cmd=0, eef_quat=单位四元数 (w=1), eef_pos=当前值
            # 说明: 全 0 动作会让两段四元数变成零范数而崩溃，
            #       故把 w 分量 (idx 5 / idx 12) 置 1，构造合法的单位四元数。
            _obs = robot.get_observation()
            _safe_action = np.zeros(action_dim, dtype=np.float32)
            # 左腕四元数 idx [5:9], 右腕四元数 idx [12:16], 设置为合法值
            _safe_action[5] = 1.0   # left_eef_quat w=1
            _safe_action[12] = 1.0  # right_eef_quat w=1
            try:
                robot.execute(_safe_action)
                # 检查 terminated 标志
                _term_val = robot._last_obs.get("terminated", False)
                _term_bool = bool(np.asarray(_term_val).any())
                logger.info("✓ [success_test] env.step() 管线: terminated=%s", _term_bool)
                # 两条判据取"或": 只要有一条为真即认为 success 条件可达。
                _success = _success or _term_bool
            except Exception as _e:
                # 非致命: 完整管线失败不影响已经拿到的直接判定结果，
                # 只降级为警告，便于区分"任务判据错"与"控制链有 bug"。
                logger.warning("⚠ env.step 管线异常 (非致命): %s", _e)

            # 输出结构与 lerobot 版本保持字段一致(success/truncated)，
            # 便于同一套结果解析脚本处理两个入口。
            results = [{"success": bool(_success), "truncated": bool(_truncated)}]
            logger.info("✓ 实验结果: %s", json.dumps(results, indent=2))
        else:
            # 常规路径: 交给 ControlLoop 跑完整闭环。
            results = loop.run(args.instruction)
            logger.info("✓ 实验结果: %s", json.dumps(results, indent=2))
    except Exception as e:
        # 顶层兜底: 记录完整堆栈；不重新抛出，保证 finally 能清理仿真。
        logger.exception("✗ 错误: %s", e)
    finally:
        # 清理顺序: 先关环境(释放 PhysX/场景)，再关 app(释放 Kit/渲染)。
        # 顺序颠倒会导致 Kit 关闭时仍在引用已销毁的场景对象。
        env.close()
        sim_app.close()
        logger.info("■ 仿真关闭完成。")


def _import_pose():
    """懒加载 Pose 类 (需要 Isaac Sim 启动后才能导入)。

    为什么不在模块顶层 import: isaaclab_arena.utils.pose 会间接 import
    Isaac Sim 的模块，在 AppLauncher 启动前导入会因 Kit 未初始化而失败。
    因此把它推迟到 main() 内部、AppLauncher 之后再调用。

    返回:
        Pose 类（用于 set_initial_pose(position_xyz=..., rotation_wxyz=...)）。
    """
    from isaaclab_arena.utils.pose import Pose
    return Pose


if __name__ == "__main__":
    main()