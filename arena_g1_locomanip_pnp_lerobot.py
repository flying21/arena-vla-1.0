#!/usr/bin/env python3
"""麻雀虽小智能科技（武汉）有限公司
══════════════════════════════════════════════════════════════════════════
  ARENA 0.1.0 — G1 移动操作 PnP 的 LeRobot 兼容封装
══════════════════════════════════════════════════════════════════════════

【设计目标】
  将 arena_g1_locomanip_pnp.py 中已验证的四层 Monkey-Patch
  (URDF/WBC/IK/Quat) 重新组织为 LeRobot 生态兼容的 Gymnasium 接口，
  使 VLA 策略（如 SmolVLA）可以像标准 LeRobot 环境一样进行训练和评估。

【架构层级（从底层到上层）】
  第 0 层: Monkey-Patch 栈
    ├── URDF 重定向 (Nucleus → GitHub Unitree)
    ├── WBC 策略 Stub (ONNX 模型占位)
    ├── PINK IK 求解器 (osqp → quadprog)
    └── 四元数归一化 (Mock 安全)
  第 1 层: Isaac Sim 引擎 (AppLauncher)
    └── headless / GUI 模式切换
  第 2 层: Arena 三核心
    ├── Scene: galileo_locomanip + brown_box + blue_sorting_bin
    ├── Embodiment: g1_wbc_pink (Unitree G1 + WBC Withen Body Control)
    └── Task: G1LocomanipPickAndPlaceTask
  第 3 层: ArenaEnvBuilder → ManagerBasedRLEnv
    └── orchestrate() + compose_manager_cfg() + 3 项修复
  第 4 层: IsaacLabEnvWrapper → Gymnasium AsyncVectorEnv
    ├── reset() 返回 (obs, info) — LeRobot 标准格式
    ├── step(action) 返回 (obs, reward, terminated, truncated, info)
    └── _get_success() 从 TerminationManager 读取 success 标志
  第 5 层: 使用接口
    ├── 方法 A: 直接 Python 调用
    │   python arena_g1_locomanip_pnp_lerobot.py --success_test
    └── 方法 B: 通过 lerobot-eval 命令行
        lerobot-eval --env.type=isaaclab_arena \
                     --env.hub_path=nvidia/isaaclab-arena-envs \
                     --env.environment=g1_locomanip_pnp \
                     --env.embodiment=g1_wbc_pink \
                     --policy.path=your_policy \
                     --eval.batch_size=1

【四层 Monkey-Patch 说明】
  由于在线资源不可达，本脚本在 Isaac Sim 启动后自动应用以下修复:
  1. 资产层: retrieve_file_path → 将 Nucleus URL 重定向到 GitHub Unitree 本地目录
  2. WBC 策略层: G1HomiePolicyV2 → _g1_wbc_stub.py 的固定站立策略
  3. PINK IK 层: qpsolvers 4.x 的 osqp 求解器 → quadprog
  4. Mock 安全层: 随机四元数零范数修复

本脚本在整体架构中的位置
════════════════════════
  本文件是【LeRobot / Gymnasium 兼容封装】入口：与同目录的
  arena_g1_locomanip_pnp.py 共享同一套场景坐标、接近度阈值与四层
  monkey-patch，但对外形态完全不同：

    ┌────────────────────┬──────────────────────────┬────────────────────────────┐
    │ 维度               │ pnp.py                    │ 本文件 (pnp_lerobot.py)     │
    ├────────────────────┼──────────────────────────┼────────────────────────────┤
    │ 组织形态           │ main() 脚本 + argparse    │ G1LocomanipPnPLeRobotEnv 类 │
    │ 闭环驱动           │ arena.client.ControlLoop  │ 调用方自己写 for 循环        │
    │ 接口契约           │ 自定义 G1LocomanipRobot   │ Gymnasium reset/step        │
    │ 典型用途           │ 人工调试 / 单次实验       │ lerobot-eval / VLA 训练评估 │
    │ success 快速验证   │ --success_test            │ --success_test / run_success_test() │
    └────────────────────┴──────────────────────────┴────────────────────────────┘

  同步维护约束: 两处坐标 (0.5785, 0.18, 0.0707) / (-0.2450, -1.6272, -0.2641)
  与阈值 (0.260/0.130/0.150) 必须一致，改一处必须同步另一处，否则两个入口
  的实验结果不可比。

依赖哪些外部系统
════════════════
  - NVIDIA Isaac Sim 5.1 + Isaac Lab 0.47.2（AppLauncher、ManagerBasedRLEnv、
    isaaclab.utils.assets.retrieve_file_path）。
  - isaaclab_arena 0.1.0（AssetRegistry / Scene / IsaacLabArenaEnvironment /
    ArenaEnvBuilder / G1LocomanipPickAndPlaceTask）。
  - isaaclab_arena_g1（G1 全身控制器：WBC 下肢策略 + PINK IK 上肢控制）。
  - 第三方库：pink、qpsolvers、numpy、torch；HTTP 后端还需 json_numpy。
  - 本仓库运行时包：arena.backends / arena.config / arena.types /
    arena.adapter / arena.client（仅在闭环演示与 HTTP 后端中用到）。
  - 本仓库辅助模块：_g1_wbc_stub.py。

运行前提
════════
  1. 已激活含 Isaac Lab 的 conda 环境，且 `arena` 包可导入；
  2. 已执行 download_g1_wbc_assets.py，使
     /tmp/unitree_ros_g1/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf 存在；
  3. 有可用 NVIDIA GPU；headless=False 时还需 X11 DISPLAY；
  4. 注意 _patch_source_files() 会尝试改写一个硬编码的第三方源文件路径，
     该路径不存在时静默跳过（见该函数说明）。

已知边界（诚实声明）
══════════════════
  - 下肢由 _g1_wbc_stub.py 的固定站立姿态驱动，机器人【不会行走】；
    mock 闭环下 success 恒为 false 属预期，只证明链路可跑通。
  - 相机被强制关闭，观测里的图像是占位张量，不可用于真实视觉策略。

══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse  # 命令行参数解析
import json      # JSON 序列化
import logging   # 日志输出
import os        # 文件路径
import sys       # sys.path 操作
from typing import Any, Dict, Optional

import numpy as np
import torch

# ── 日志配置 ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# 本模块单独用 "arena_lerobot" 这个名字，便于在两个入口混跑时区分日志来源。
logger = logging.getLogger("arena_lerobot")


# ═══════════════════════════════════════════════════════════════════
# 第 0 层: Monkey-Patch 栈 — 修复在线资源不可达的四层缺陷
#   ① 资产重定向 (Nucleus → 本地 Unitree 克隆)
#   ② WBC 下肢策略 Stub (ONNX 权重不可达)
#   ③ PINK IK 求解器 (osqp → quadprog)
#   ④ Mock 动作四元数归一化 (实际实现位于 build_mock_client)
# ═══════════════════════════════════════════════════════════════════

def _patch_assets():
    """
    第 1 层修复: 资产重定向
    ┌─────────────────────────────────────────────────────────────┐
    │ 缺陷: G1 WBC 资产 URL 指向 NVIDIA 内部 Nucleus 服务器       │
    │       omniverse://isaac-dev.ov.nvidia.com/...               │
    │ 修复: monkey-patch retrieve_file_path(),                     │
    │       将 URDF 和 mesh 目录重定向到 Unitree GitHub 本地克隆  │
    │ URDF: /tmp/unitree_ros_g1/robots/g1_description/            │
    │        g1_29dof_with_hand_rev_1_0.urdf                      │
    └─────────────────────────────────────────────────────────────┘

    为什么用 GitHub 克隆: NVIDIA 的 Nucleus 地址只在内部网络可达，
    离线环境必须自备资产；Unitree 官方 unitree_ros 仓库提供了同一台 G1 的
    URDF 与 meshes，是唯一公开且完整的等价来源，由
    download_g1_wbc_assets.py 预先克隆到 /tmp。

    参数: 无。
    返回: 无。
    失败行为: 本地 URDF 缺失时不抛异常，只把日志标记为 "缺失!"，
              真实失败延后到资产加载阶段以解析错误形式出现。
    """
    _g1_urdf = "/tmp/unitree_ros_g1/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf"
    _g1_pkg_dir = "/tmp/unitree_ros_g1/robots/g1_description"

    import isaaclab.utils.assets as _assets_mod

    # 保存原始函数引用
    # (命中 Nucleus 前缀以外的路径仍要走原始逻辑，例如其它基础资产)
    _original_retrieve = _assets_mod.retrieve_file_path
    _NUCLEUS_PREFIX = "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets"

    def _patched_retrieve(path: str, download_dir=None, force_download=True) -> str:
        """代理 retrieve_file_path: 拦截 Nucleus URL 并返回本地文件路径。

        参数:
            path: 原始资产路径（Nucleus URL 或普通路径）。
            download_dir: 透传给原始实现的下载目录。
            force_download: 透传给原始实现的强制下载标志。

        返回:
            命中前缀时返回本地绝对路径；否则返回原始实现的返回值。
        """
        if path.startswith(_NUCLEUS_PREFIX):
            # 区分 URDF 文件和资产目录
            # 判据: basename 是否含 "urdf" 子串。
            #   命中  → 单文件模型描述，返回具体 URDF 路径；
            #   未命中 → 目录型资产 (meshes / robot_model / wbc_policy)，
            #            统一返回包目录，让 meshdir="meshes" 的相对解析生效。
            if "urdf" in os.path.basename(path).lower():
                logger.info("  → 使用本地 G1 URDF: %s", _g1_urdf)
                return os.path.abspath(_g1_urdf)
            logger.info("  → 使用本地 G1 包目录: %s", _g1_pkg_dir)
            return os.path.abspath(_g1_pkg_dir)
        return _original_retrieve(path, download_dir=download_dir, force_download=force_download)

    # 只替换模块属性：对已经 `from ... import retrieve_file_path` 绑定别名的
    # 调用者无效，故这是"尽力而为"的补丁。
    _assets_mod.retrieve_file_path = _patched_retrieve
    _ok = os.path.isfile(_g1_urdf)
    logger.info("✓ [Patch 1/4] G1 WBC 资产已重定向 (URDF: %s)", "OK" if _ok else "缺失!")


def _patch_wbc_policy():
    """
    第 2 层修复: WBC 策略 Stub
    ┌─────────────────────────────────────────────────────────────┐
    │ 缺陷: g1_homie_policy_v2.pt (ONNX 模型) 不可达              │
    │       WBC 下肢策略需要 ONNX Runtime 推理                     │
    │ 修复: 用 _g1_wbc_stub.py 的 G1WBCLowerBodyStub 替换         │
    │       G1HomiePolicyV2, 返回恒定的站立姿态                    │
    │ 效果: 腿部固定站立, 上肢由 PINK IK 单独控制                 │
    └─────────────────────────────────────────────────────────────┘

    实现方式: 直接改写 Arena 工厂模块的类属性
    (wbc_policy_factory.G1HomiePolicyV2)，Arena 内部查表构造策略时
    就会拿到 Stub，无需改动 SDK 源码。

    接口边界: Stub 输出恒定站立姿态，机器人【不会行走】——导航子目标
              (navigate_cmd) 不会被真正执行。它保证环境能构建、任务链路可验证，
              但 mock 下 success=false 是预期行为，不代表任务已实现。

    参数: 无。
    返回: 无（apply_patch() 的布尔结果被丢弃）。
    失败行为: apply_patch() 内部只记录警告、不抛异常；若它失败，
              故障会以"后续不相关的报错"形式出现，排障时应先搜索
              日志中的 "WBC Stub patch 失败"。
    """
    from _g1_wbc_stub import apply_patch
    apply_patch()
    logger.info("✓ [Patch 2/4] WBC 下肢策略已替换为本地 Stub (固定站立姿态)")


def _patch_pink_ik():
    """
    第 3 层修复: PINK IK 求解器
    ┌─────────────────────────────────────────────────────────────┐
    │ 缺陷: qpsolvers 4.13.0 不再内置 osqp 求解器                  │
    │       g1_wbc_upperbody_controller.py 硬编码 solver="osqp"    │
    │ 修复: 双重 monkey-patch:                                    │
    │       (a) 修改源文件 solver="osqp" → solver="quadprog"      │
    │       (b) 运行时 patch pink.solve_ik.solve_ik               │
    └─────────────────────────────────────────────────────────────┘

    为什么 qpsolvers 4.x 里 osqp 不可用: qpsolvers 从 1.x 演进到 4.x 后
    改为"可选后端"设计，osqp 不再随包安装/导出；而 Arena 的 G1 上肢控制器
    把求解器名校验放在 solve_ik 里，硬编码 "osqp" 会直接抛
    "solver not found"。quadprog 解决同一类凸 QP，且在 4.x 仍可用，
    故把求解器名改写为 quadprog。

    为什么需要"源文件 + 运行时"双重 patch:
      - 源文件改写解决"模块顶层就 import 并解析了求解器"的场景，
        但要求第三方文件路径可写、且必须在导入该模块之前完成；
      - 运行时改写解决"源文件不可写 / 路径不存在 / 模块已被导入"的场景，
        它是唯一对已绑定别名生效的手段。
      两者互补：任一环境条件下总有一层能兜住。

    参数: 无。
    返回: 无。
    失败行为: pink 顶层别名不存在时静默忽略（Arena 版本差异下可能没有该导出）。
    """
    import importlib as _il

    # (a) 源文件级别: 已在 _patch_source_files() 中独立处理

    # (b) 运行时级别: pink.solve_ik 的模块引用
    _pink_ik = _il.import_module("pink.solve_ik")
    _orig = _pink_ik.solve_ik

    def _patched_solve(*args, **kwargs):
        """包装 solve_ik: 强制 solver="quadprog" 替代不可用的 osqp。

        参数: 与 pink.solve_ik 一致（*args/**kwargs 透传）。
        返回: pink.solve_ik 的原始返回值（IK 解）。
        """
        kwargs.setdefault("solver", "quadprog")
        if kwargs.get("solver") == "osqp":
            kwargs["solver"] = "quadprog"
        return _orig(*args, **kwargs)

    _pink_ik.solve_ik = _patched_solve

    # 同时 patch pink 包的顶层别名
    import pink as _pink_mod
    try:
        setattr(_pink_mod, "solve_ik", _patched_solve)
    except Exception:
        pass

    logger.info("✓ [Patch 3/4] PINK IK 求解器已切换为 quadprog (osqp 不可用)")


def _patch_quaternion():
    """
    第 4 层修复: Mock 动作的四元数归一化
    ┌─────────────────────────────────────────────────────────────┐
    │ 缺陷: mock 后端的 23 维随机动作中, 第 5-9 维 (left_wrist)   │
    │       和第 12-16 维 (right_wrist) 是四元数 [w,x,y,z]        │
    │       随机值可能全为 0, 导致 scipy Rotation.from_quat()     │
    │       抛出 "Found zero norm quaternions" 错误               │
    │ 修复: 在 InProcessClient.infer() 中做归一化,                │
    │       范数 ≤ 1e-6 → 设置为单位四元数 [1,0,0,0]              │
    │ 说明: 仅在 mock 模式下生效, 不影响真实策略推理               │
    └─────────────────────────────────────────────────────────────┘

    本函数只是"占位 + 日志"，真正的归一化逻辑必须写在动作生成处
    （build_mock_client 内的 InProcessClient.infer），因为补丁需要拿到
    每条动作向量的上下文才能就地改写 [5:9]/[12:16] 两段。

    参数: 无。
    返回: 无（不产生任何副作用）。
    """
    # 此修复需要 InProcessClient 的上下文, 在 build_mock_client() 内实现
    # 这里仅记录修复说明
    logger.info("✓ [Patch 4/4] Mock 四元数归一化修复已就绪 (在 build_mock_client 中)")


def _patch_source_files():
    """
    源文件级别的静态修复 — 直接修改第三方的 .py 文件。

    修复项:
      - g1_wbc_upperbody_controller.py L179: solver="osqp" → solver="quadprog"
    
    运行时 patch (runtime monkey-patch) 已在 _patch_pink_ik() 中处理。

    为什么需要它: 该控制器模块在导入期就会解析求解器名，仅靠运行时改写
    solve_ik 覆盖不到"模块导入时已完成解析"的情况，因此需要在导入前把
    源文件里的字面量改掉。

    参数: 无。
    返回: 无。
    失败行为（三点，均为静默降级）:
      1. 路径是硬编码的他人工作目录
         (/home/rq/文档/project/unitree_G1D/...)，在其它机器上必然不存在，
         此时 os.path.isfile() 为假，函数直接返回、不报错；
      2. 文件存在但内容里没有 'solver="osqp"' 字样时不做任何写入；
      3. 本函数以普通文本 replace 改写第三方源文件，属于有副作用的操作
         （会永久修改外部文件），且没有备份。
    """
    _controller_file = (
        "/home/rq/文档/project/unitree_G1D/Arena_0_1_0/IsaacLab-Arena/"
        "isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/"
        "g1_wbc_upperbody_ik/g1_wbc_upperbody_controller.py"
    )
    if os.path.isfile(_controller_file):
        with open(_controller_file, "r") as f:
            _content = f.read()
        if 'solver="osqp"' in _content:
            _content = _content.replace('solver="osqp"', 'solver="quadprog"')
            with open(_controller_file, "w") as f:
                f.write(_content)
            logger.info("  → 源文件修复: solver=osqp → solver=quadprog")


def _apply_all_patches():
    """按依赖顺序 (资产→WBC策略→IK求解器→四元数) 应用全部四层修复。

    顺序约束: _patch_source_files() 必须最早——它要在目标模块被 import 之前
    落盘，否则改写已经来不及；其余三层在 Isaac Sim 启动后、资产加载前完成即可。

    参数: 无。
    返回: 无。
    """
    _patch_source_files()   # 静态文件修复 (最早, 在导入模块前)
    _patch_assets()          # Layer 1: 资产
    _patch_wbc_policy()      # Layer 2: WBC
    _patch_pink_ik()         # Layer 3: IK
    _patch_quaternion()      # Layer 4: 四元数 (记录日志)
    logger.info("══ 四层 Monkey-Patch 已全部应用 ══")


# ═══════════════════════════════════════════════════════════════════
# 环境创建 — 场景构建 / 配置生成与修复 / 环境实例化
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# 第 5 层: LeRobot 兼容环境类
# ═══════════════════════════════════════════════════════════════════

class G1LocomanipPnPLeRobotEnv:
    """
    G1 移动操作箱子抓取与放置 (Loco-Manipulation PnP) 的 LeRobot 兼容封装。

    核心特征:
      - 标准 Gymnasium AsyncVectorEnv 接口 (通过 IsaacLabEnvWrapper)
      - 内置四层 Monkey-Patch 自动修复
      - 支持 --success_test 模式 (验证 PnP 成功终止条件)
      - 支持 lerobot-eval 命令行调用

    动作空间 (23 维):
      ┌─────────────────────────────────────────────────────┐
      │ 索引   │ 名称                  │ 维度 │ 说明        │
      ├─────────────────────────────────────────────────────┤
      │ 0-1    │ if_navigate / nav_vel  │ 2    │ 导航命令    │
      │ 2-4    │ left_eef_pos          │ 3    │ 左手 EEF 位置│
      │ 5-8    │ left_eef_quat         │ 4    │ 左手 EEF 四元数│
      │ 9-11   │ right_eef_pos         │ 3    │ 右手 EEF 位置│
      │ 12-15  │ right_eef_quat        │ 4    │ 右手 EEF 四元数│
      │ 16-18  │ navigate_cmd          │ 3    │ 导航子目标  │
      │ 19     │ base_height_cmd       │ 1    │ 底座高度    │
      │ 20-22  │ torso_orientation_cmd │ 3    │ 躯干姿态 RPY│
      └─────────────────────────────────────────────────────┘

    观测空间 (policy 组):
      - robot_joint_pos: (43,) 关节位置
      - robot_joint_vel: (43,) 关节速度
      - left/right_wrist_pose: (4,4) 腕部位姿 (pelvis 坐标系)
      - left/right_eef_pos/quat: EEF 位姿
      - body_eef_pos/quat: 躯干位姿
      - robot_pos/quat: 机器人世界位姿

    成功条件 (objects_in_proximity):
      - 箱子中心进入分拣箱 ±26×13×15cm 范围内

    ⚠ 文档与实现不一致（保留原状，仅在此说明）:
      上面的"核心特征/第 4 层"沿用了设计文档的写法，声称本类通过
      IsaacLabEnvWrapper 暴露 Gymnasium 接口；但实际实现
      （见 _build_gymnasium_env()）**并没有使用 IsaacLabEnvWrapper**，
      而是直接持有并调用 self._raw_env（ManagerBasedRLEnv 本身已返回
      Gymnasium 兼容的 (obs, info) / 5 元组）。因此：
        - 不存在 AsyncVectorEnv 包装，reset/step 是同步单环境语义；
        - 也没有环境方法 _get_success()，"success" 由 step() 自行往
          info["final_info"]["is_success"] 里填（其取值等于 terminated，
          见 step() 的说明）；
        - 若按 lerobot-eval 的 isaaclab_arena 环境注册方式接入，需要额外
          适配层，本文件只提供可被其调用的类。

    生命周期: __init__ 会启动 Isaac Sim 并创建环境，务必用 close() 或
              上下文管理器（with）释放，否则 Kit/PhysX 不会退出。
    """

    def __init__(
        self,
        headless: bool = True,
        enable_cameras: bool = False,
        object_name: str = "brown_box",
        embodiment_name: str = "g1_wbc_pink",
        episode_length: int = 500,
        seed: int = 42,
        success_test: bool = False,
    ):
        """
        初始化 LeRobot 兼容的 G1 移动操作 PnP 环境。

        参数:
            headless: 是否无头模式 (True=不弹出渲染窗口)
            enable_cameras: 是否启用相机渲染 (当前禁用: camera data 0 dim workaround)
            object_name: 要抓取的物体名 (默认: brown_box)
            embodiment_name: 机器人实施例名 (默认: g1_wbc_pink)
            episode_length: episode 最大步数
            seed: 随机种子
            success_test: 是否进入 success 验证模式

        返回: 无（构造出实例；action_dim 属性在构造末尾被赋值）。

        失败行为: 不吞异常——AppLauncher 启动失败、patch 失败、场景/环境
                  构建失败都会直接抛出，调用方需自行捕获（main() 里就是这么做的）。
                  注意：若 AppLauncher 已启动而后续步骤抛错，Isaac Sim 不会被
                  自动关闭，需调用 close() 或依赖 finally。
        """
        import argparse as _argparse

        self.headless = headless
        self.enable_cameras = enable_cameras
        self.object_name = object_name
        self.embodiment_name = embodiment_name
        self.episode_length = episode_length
        self.seed = seed
        self.success_test = success_test

        # ── 第 1 层: 启动 Isaac Sim 仿真引擎 ────────────────────────
        # 必须是本类中最早的实质动作：所有 isaaclab_arena 导入、资产注册、
        # PhysX 场景创建都依赖 Kit 已经初始化。
        from isaaclab.app import AppLauncher

        logger.info("▶ 启动 Isaac Sim (headless=%s, cameras=%s)...", headless, enable_cameras)
        self.sim_app = AppLauncher(
            headless=headless,
            enable_cameras=enable_cameras,
        ).app

        # ── 应用全部 Monkey-Patch ──────────────────────────────────
        # 时机: 在 Kit 就绪之后、任何资产/策略/IK 模块被导入使用之前。
        _apply_all_patches()

        # ── 第 2 层: 构建 Arena 三核心 ──────────────────────────────
        self._build_arena_environment()

        # ── 第 3-4 层: ManagerBasedRLEnv + IsaacLabEnvWrapper ───────
        self._build_gymnasium_env()

        # ── 获取动作维度 ───────────────────────────────────────────
        # 从真实环境中读维度，供 build_mock_client / run_success_test 构造
        # 宽度正确的安全动作向量。
        self.action_dim = int(self._raw_env.action_space.shape[-1])
        logger.info("✓ LeRobot 兼容 G1 PnP 环境已创建 — action_dim=%d", self.action_dim)

    def _build_arena_environment(self):
        """
        构建 Arena 三核心 (Scene + Embodiment + Task)。

        复用 arena_g1_locomanip_pnp.py 的场景组合逻辑:
          - Galileo Loco-Manipulation 操作场景
          - 棕色箱子 (brown_box) @ (0.5785, 0.18, 0.0707)
          - 蓝色分拣箱 (blue_sorting_bin) @ (-0.245, -1.627, -0.264)
          - Unitree G1 @ (0.0, 0.18, 0.0)
          - 任务: G1LocomanipPickAndPlaceTask (episode_length=20s)

        参数: 无（使用 self.object_name / self.embodiment_name / self.seed）。
        返回: 无；结果写入实例属性：
              self._arena_env / self._cli_args / self._pick_up_object /
              self._blue_sorting_bin / self._embodiment。
        失败行为: 资产名不存在时 AssetRegistry 会抛错；本函数不捕获。
        """
        import argparse as _argparse
        from isaaclab_arena.assets.asset_registry import AssetRegistry
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import (
            G1LocomanipPickAndPlaceTask,
        )
        from isaaclab_arena.utils.pose import Pose

        # 这里直接 import Pose 而不是懒加载：本方法只在 AppLauncher 启动后调用，
        # 因此不存在"Kit 未初始化"的导入风险。
        asset_registry = AssetRegistry()

        # ── 背景场景 ───────────────────────────────────────────────
        logger.info("▶ 加载 Galileo Loco-Manipulation 操作场景...")
        background = asset_registry.get_asset_by_name("galileo_locomanip")()

        # ── 抓取物体 (棕色箱子) ─────────────────────────────────────
        logger.info("▶ 加载棕色箱子 (brown_box)...")
        pick_up_object = asset_registry.get_asset_by_name(self.object_name)()
        # rotation_wxyz=(0,0,1,0) → (w,x,y,z) 下为绕 Y 轴 180°，
        # 让箱子开口朝向 G1（与 pnp.py 保持一致）。
        pick_up_object.set_initial_pose(
            Pose(position_xyz=(0.5785, 0.18, 0.0707), rotation_wxyz=(0.0, 0.0, 1.0, 0.0))
        )

        # ── 目标物体 (蓝色分拣箱) ───────────────────────────────────
        logger.info("▶ 加载蓝色分拣箱 (blue_sorting_bin)...")
        blue_sorting_bin = asset_registry.get_asset_by_name("blue_sorting_bin")()
        # 该坐标同时被 run_success_test() 的 _bin_init 常量引用，改一处须同步另一处。
        blue_sorting_bin.set_initial_pose(
            Pose(position_xyz=(-0.2450, -1.6272, -0.2641), rotation_wxyz=(0.0, 0.0, 0.0, 1.0))
        )

        # ── 机器人实施例 ──────────────────────────────────────────
        logger.info("▶ 加载宇树 G1 WBC Pink 实施例 (全身控制 + PINK IK)...")
        # enable_cameras 透传给 embodiment：关闭相机时不应创建相机资产，
        # 否则 syntheticdata 仍会初始化并触发 0 维崩溃。
        embodiment = asset_registry.get_asset_by_name(self.embodiment_name)(
            enable_cameras=self.enable_cameras
        )
        # 单位四元数 (w=1) → 朝向 +X，正对操作台。
        embodiment.set_initial_pose(
            Pose(position_xyz=(0.0, 0.18, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0))
        )

        # ── 组合场景 ──────────────────────────────────────────────
        assets = [background, pick_up_object, blue_sorting_bin]
        scene = Scene(assets=assets)

        # ── 定义任务 ──────────────────────────────────────────────
        # 接近度阈值由 SDK 任务默认值提供；此处只覆盖 episode 时长 20s。
        task = G1LocomanipPickAndPlaceTask(
            pick_up_object,
            blue_sorting_bin,
            background,
            episode_length_s=20.0,
        )
        logger.info("▶ 任务: G1LocomanipPickAndPlaceTask (阈值: X±26cm Y±13cm Z±15cm, 超时=20s)")

        # 保存资产引用 (success_test 需要)
        # 保留 Python 引用还有一层作用：避免资产对象被 GC 提前回收。
        self._pick_up_object = pick_up_object
        self._blue_sorting_bin = blue_sorting_bin
        self._embodiment = embodiment

        # ── 构建 Arena 环境 ──────────────────────────────────────
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

        self._arena_env = IsaacLabArenaEnvironment(
            name="galileo_g1_locomanip_pick_and_place",
            embodiment=embodiment,
            scene=scene,
            task=task,
        )

        # ── 组装 argparse Namespace (传给 ArenaEnvBuilder) ──────────
        # ArenaEnvBuilder 按属性名读取参数，因此这里手工造一个 Namespace，
        # 而不是真的解析命令行。
        # 注意: enable_cameras 用的是 self.enable_cameras（默认 False），
        # 而 pnp.py 的同一个 Namespace 里写的是 True —— 两个入口此处不一致。
        self._cli_args = _argparse.Namespace(
            headless=self.headless,
            enable_cameras=self.enable_cameras,
            device="cuda:0",
            num_envs=1,
            disable_fabric=True,
            seed=self.seed,
            task="pick up the brown box and place it in the blue bin",
            embodiment=self.embodiment_name,
            object=self.object_name,
            teleop_device=None,
            mimic=False,
            video=False,
            video_length=0,
            video_interval=0,
        )

    def _build_gymnasium_env(self):
        """
        构建标准 Gymnasium 环境。

        通过 ArenaEnvBuilder 生成 ManagerBasedRLEnvCfg,
        创建原始的 Isaac Lab 批量 GPU 环境,
        再包装为 IsaacLabEnvWrapper (Gymnasium AsyncVectorEnv)。

        参数: 无（读取 self._arena_env / self._cli_args）。
        返回: 无；结果写入 self._raw_env。
        失败行为: 配置或场景构建失败时 ManagerBasedRLEnv 会抛错，本函数不捕获。
        """
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

        # ── 使用 ArenaEnvBuilder 编排并生成配置 ─────────────────────
        # orchestrate(): 组装资产与 manager 项；
        # compose_manager_cfg(): 产出纯配置对象 ManagerBasedRLEnvCfg。
        builder = ArenaEnvBuilder(self._arena_env, self._cli_args)
        builder.orchestrate()
        cfg = builder.compose_manager_cfg()

        # ── 3 项配置修复 ───────────────────────────────────────────
        # 修复 1: 移除 last_action 观测 (0 维张量导致崩溃)
        #   单环境 + 关闭相机后 last_action 张量退化为 0 维，拼接观测时 shape 报错；
        #   本任务不需要它，置 None 让 manager 跳过（None 而非删除，因为
        #   ObservationGroupCfg 的属性必须在位）。
        if hasattr(cfg.observations.policy, "actions"):
            cfg.observations.policy.actions = None
        # 修复 2: 禁用相机观测 (Isaac Sim 5.1 camera data 0 dim workaround)
        #   相机整体关闭（enable_cameras=False），保留 camera_obs 会让 manager
        #   去读不存在的相机 buffer；置 None 跳过。
        if hasattr(cfg.observations, "camera_obs"):
            cfg.observations.camera_obs = None
        # 修复 3: 单环境模式
        #   强制 num_envs=1：单机器人调试场景，既省显存也避免终止信号聚合歧义
        #   （LeRobot 侧按 batch=1 消费 terminated/truncated）。
        cfg.scene.num_envs = 1

        # ── 创建原始环境 ───────────────────────────────────────────
        self._raw_env = ManagerBasedRLEnv(cfg=cfg)
        # 禁止 Isaac Sim 自动关闭 (我们手动管理生命周期)
        # 否则 app 可能在 self._raw_env 仍被引用时被销毁。
        self._raw_env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

        # ── ManagerBasedRLEnv 已支持 Gymnasium 兼容返回格式 ─────────
        # reset() → (obs, info)  是一个 2-元组
        # step()  → (obs, reward, terminated, truncated, info)  是 5-元组
        # 无需额外的 IsaacLabEnvWrapper，直接使用 _raw_env
        #
        # ⚠ 与类文档字符串/设计文档的差异（仅记录，不修改代码）:
        #   上面两句说明"实际实现没有使用 IsaacLabEnvWrapper"，
        #   而类 docstring 与文档里的"第 4 层: IsaacLabEnvWrapper →
        #   Gymnasium AsyncVectorEnv"是设计文档的旧描述。二者不一致：
        #     - 本类没有包装成 AsyncVectorEnv，reset/step 是同步的单环境调用；
        #     - 也没有 _get_success() 方法，success 由 step() 自行写入
        #       info["final_info"]["is_success"]。
        #   因此若按 AsyncVectorEnv 语义（例如自动批处理、异步 reset）来使用，
        #   会与实现不符。

    # ── 公有接口 (Gymnasium 兼容) ─────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        """
        重置环境并返回初始观测。

        参数:
            seed: 可选随机种子；为 None 时使用构造时的 self.seed。

        返回:
            (obs, info) — LeRobot 标准 Gymnasium 格式

        失败行为: 底层 reset 抛出的异常不捕获，直接向上传播。
        兼容性说明: 若底层返回的不是 2 元组（旧版单元素/纯 obs），
                    则退化为 (result, 伪造 info)，其中
                    final_info.is_success 被置为 [False]——这是一个占位值，
                    并不代表真实成功状态。
        """
        _seed = seed if seed is not None else self.seed
        result = self._raw_env.reset(seed=_seed)
        # ManagerBasedRLEnv.reset() 返回 (obs, info) 或 (obs,) 取决于版本
        if isinstance(result, tuple) and len(result) == 2:
            return result[0], result[1]
        # 单元素→补空 info
        # 注意: 这里返回的 result 可能是 (obs,)，调用方需要自行容忍该形状差异。
        _info = {"final_info": {"is_success": np.array([False])}}
        return result, _info

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        执行一步仿真 — ManagerBasedRLEnv 原生 Gymnasium 兼容接口。

        参数:
            action: numpy 数组, shape=(1, action_dim) 或 (action_dim,)

        返回:
            (obs, reward, terminated, truncated, info)
            - info["final_info"]["is_success"]: 任务是否成功 (布尔值)

        实现说明:
          - 动作先 reshape 成 (1, -1) 以匹配 num_envs=1 的批契约，
            再转到环境所在设备（cuda）并包在 inference_mode 下执行；
          - 5 元组分支是标准路径；4 元组分支为旧版兼容，把 done 同时
            当作 terminated 与 truncated 返回；
          - info 里填入的 is_success 直接取 terminated，因此
            object_dropped / 超时等失败终止同样会被标记为 is_success=True
            （潜在的语义混淆，见最终报告，未修改代码）。

        失败行为: 任何异常都会被 logger.error 记录后 **重新抛出**（fail-fast），
                  避免调用方把"崩溃"误当作"episode 结束"。
        """
        try:
            # 转换为 GPU 张量 → ManagerBasedRLEnv.step() 期望 (batch, dim)
            _act = np.asarray(action, dtype=np.float32).reshape(1, -1)
            _act_t = torch.from_numpy(_act).to(self._raw_env.device)

            with torch.inference_mode():
                _result = self._raw_env.step(_act_t)

            # ManagerBasedRLEnv.step() 返回 5-元组: (obs, reward, terminated, truncated, info)
            if isinstance(_result, tuple) and len(_result) >= 5:
                _obs, _reward, _term, _trunc, _info = _result[:5]
                # 补充 final_info (LeRobot 需要的 is_success)
                # ⚠ 这里用 terminated 充当 is_success：成功与失败终止
                #   （例如 object_dropped / 箱子掉地）无法区分。
                _is_success = _term.cpu().numpy().astype(bool)
                _info["final_info"] = {"is_success": _is_success}
                return _obs, _reward, _term, _trunc, _info
            # 回退: 旧版 4-元组 (obs, reward, done, info)
            _obs, _reward, _done, _info = _result
            _done_np = _done.cpu().numpy().astype(bool)
            return _obs, _reward, _done_np, _done_np, _info
        except Exception as e:
            logger.error("✗ step() 执行失败: %s", e)
            raise

    def close(self):
        """安全关闭环境和 Isaac Sim 仿真。

        参数/返回: 无。
        关闭顺序: 先 _raw_env.close()（释放场景与 PhysX），再 sim_app.close()
                  （释放 Kit/渲染）。顺序颠倒会在 Kit 关闭时引用已销毁的场景。
        失败行为: 两步各自的异常都被吞掉——即便环境已关闭或 app 已退出，
                  本方法也不会抛错，保证 finally 块安全。
        """
        try:
            self._raw_env.close()
        except Exception:
            pass
        try:
            self.sim_app.close()
        except Exception:
            pass

    @property
    def unwrapped(self):
        """返回原始 ManagerBasedRLEnv (用于访问底层属性)。

        返回:
            self._raw_env.unwrapped（若存在）否则 self._raw_env。
        """
        return self._raw_env.unwrapped if hasattr(self._raw_env, 'unwrapped') else self._raw_env

    @property
    def observation_space(self):
        """底层环境的观测空间（gymnasium.spaces.Dict），透传给 LeRobot 侧做校验。"""
        return self._raw_env.observation_space

    @property
    def action_space(self):
        """底层环境的动作空间（gymnasium.spaces.Box），透传给 LeRobot 侧做校验。"""
        return self._raw_env.action_space

    def __enter__(self):
        """上下文管理器入口：返回自身，便于 `with G1LocomanipPnPLeRobotEnv() as env:`。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口：无论是否异常都调用 close()。

        返回:
            False —— 不吞异常，异常继续向上传播（只是保证先完成清理）。
        """
        self.close()
        return False

    # ── success_test 模式 ──────────────────────────────────────────

    def run_success_test(self) -> dict:
        """
        ★ 验证 PnP 任务成功终止条件的测试模式。

        实现流程:
          1. 重置环境
          2. 读取分拣箱的世界坐标
          3. 将棕箱 teleport 到分拣箱中心正上方 10cm
          4. 通过 Physical Simulator 步进 3 帧让 PhysX 传播位置
          5. 直接从 RigidObject 读取位置, 计算 ΔX/ΔY/ΔZ
          6. 与 objects_in_proximity 阈值 (X±26cm, Y±13cm, Z±15cm) 对比
          7. 尝试 env.step() 走完整终止检测管线

        参数: 无（读取 self._pick_up_object / self._blue_sorting_bin / self.action_dim）。
        返回:
            {"success": bool, "deltas": {"dx":, "dy":, "dz":}, "env_step_ok": bool}
            success = 直接距离判据 或 env.step 终止标志（逻辑或）。

        失败行为: teleport 与直接判据部分不捕获异常（失败即抛出）；
                  env.step() 管线部分单独 try/except，异常降级为 warning，
                  并把 env_step_ok 保持 False——这样"控制链有 bug"不会
                  掩盖"任务判据本身是否正确"这个待验证结论。
        """
        logger.info("▶ [success_test] 重置环境...")
        self.reset()

        # ── 获取 RigidObject 引用 ──────────────────────────────────
        # scene[name] 返回 Arena 注册的刚体句柄；通过它 teleport 与读世界位姿。
        _scene = self._raw_env.unwrapped.scene
        _pick_obj = _scene[self._pick_up_object.name]   # 棕色箱子
        _box_obj = _scene[self._blue_sorting_bin.name]  # 蓝色分拣箱

        # ── 分拣箱初始世界坐标 (blueprint 定义) ────────────────────
        # 与 _build_arena_environment() 中 set_initial_pose 的常量必须一致。
        _bin_init = np.array([-0.2450, -1.6272, -0.2641], dtype=np.float32)

        # ── Teleport: 棕箱 → 分拣箱正上方 10cm ────────────────────
        # 偏移 10cm 既落在 Z 阈值 ±0.150 之内，又给箱子留出下落余量，
        # 避免刚放入就因穿模被弹开导致 Z 向超差。
        _target = _bin_init + np.array([0.0, 0.0, 0.10])
        logger.info("▶ [success_test] Teleport: 棕箱 → (%.3f, %.3f, %.3f)", *_target)

        # 构建 7 维位姿 [x, y, z, qx, qy, qz, qw] (Isaac Sim 用 xyzw 顺序)
        # 注意与 Arena Pose 的 rotation_wxyz=(w,x,y,z) 的差异：
        # PhysX 的 root pose 通道是 xyzw，故单位旋转写成 (0,0,0,1)，
        # 首尾顺序若弄反会得到 180° 的错误姿态。
        # 这里不需要额外归一化，因为常量本身已是单位四元数。
        _pose = torch.tensor(
            [[_target[0], _target[1], _target[2], 0.0, 0.0, 0.0, 1.0]],
            device=self._raw_env.device,
            dtype=torch.float32,
        )
        _pick_obj.write_root_pose_to_sim(_pose)

        # ── PhysX 步进: 让物理引擎传播位置 ─────────────────────────
        # write_root_pose_to_sim 只写 staging buffer，需 sim.step() 才把位姿
        # 积分/广播到 data.root_pos_w 视图；单帧有时仍读到旧值，
        # 连续 3 帧是经验上稳定可见的余量（远小于 20s episode 预算）。
        # 注意: 这里直接驱动物理引擎，绕过了 manager 的 decimation 与动作/观测
        # 缓存更新，因此后续 manager 侧观测可能仍是 step 之前的内容。
        for _ in range(3):
            self._raw_env.sim.step(render=False)

        # ── 直接读取位置验证 ──────────────────────────────────────
        # 为什么手动算 ΔX/ΔY/ΔZ: 不走 TerminationManager 就没有 terminated 标志，
        # 而本段的目的正是独立验证 success 判据本身；用与 SDK 完全相同的阈值
        # (0.260/0.130/0.150) 复算，可把"任务参数写错"与"终止管理器没触发"
        # 两类问题区分开。
        _p_pick = _pick_obj.data.root_pos_w.cpu().numpy().reshape(-1)
        _p_box = _box_obj.data.root_pos_w.cpu().numpy().reshape(-1)
        _dx = abs(_p_pick[0] - _p_box[0])
        _dy = abs(_p_pick[1] - _p_box[1])
        _dz = abs(_p_pick[2] - _p_box[2])
        _sep_ok = _dx < 0.260 and _dy < 0.130 and _dz < 0.150

        logger.info("✓ 棕箱(%.3f, %.3f, %.3f) vs 分拣箱(%.3f, %.3f, %.3f)",
                    *_p_pick, *_p_box)
        logger.info("✓ ΔX=%.3f (ok<0.260) ΔY=%.3f (ok<0.130) ΔZ=%.3f (ok<0.150) → %s",
                    _dx, _dy, _dz, "SUCCESS" if _sep_ok else "FALSE")

        # ── 尝试 env.step() 管线 ───────────────────────────────────
        # 目的: 用完整管线（WBC + PINK IK + TerminationManager）再验证一次
        #       success 终止条件是否会被真正置位。
        _env_ok = False
        try:
            # 全 0 动作会让四元数段变成零范数，故把 w 分量置 1 构造合法单位四元数。
            _safe_action = np.zeros(self.action_dim, dtype=np.float32)
            _safe_action[5] = 1.0   # left_eef_quat w=1
            _safe_action[12] = 1.0  # right_eef_quat w=1
            _, _, terminated, truncated, info = self.step(_safe_action)
            _env_ok = bool(terminated.any())
            logger.info("✓ env.step() 管线: terminated=%s", _env_ok)
        except Exception as _e:
            logger.warning("⚠ env.step 管线异常 (非致命): %s", _e)

        # 与 pnp.py 的结果结构不同: 这里额外给出 deltas 与 env_step_ok，
        # 便于区分"距离判据通过"和"终止管理器置位"两个层次的结论。
        result = {
            "success": bool(_sep_ok or _env_ok),
            "deltas": {"dx": float(_dx), "dy": float(_dy), "dz": float(_dz)},
            "env_step_ok": _env_ok,
        }
        return result


# ═══════════════════════════════════════════════════════════════════
# Mock 客户端 — 安全的随机动作生成器
# ═══════════════════════════════════════════════════════════════════

def build_mock_client(env: G1LocomanipPnPLeRobotEnv):
    """
    构建 Mock 策略客户端 — 生成符合约束的安全随机动作。

    关键修复 (Patch 4/4):
      动作维度的四元数 [5:9] 和 [12:16] 需要归一化,
      避免 scipy Rotation.from_quat 收到零范数四元数。

    参数:
        env: 已创建的环境实例，用它的 action_dim 配置随机采样宽度。

    返回:
        InProcessClient 实例（进程内策略客户端，不经过 HTTP）。

    失败行为: arena.backends 导入失败或 backend 构造失败时直接抛出。
    """
    from arena.backends import build_backend
    from arena.config import ServerConfig
    from arena.types import ActionChunk

    # mock 后端：进程内随机采样，用于在没有 VLA 服务器时打通链路。
    backend = build_backend(ServerConfig(backend="mock"))
    # 注入真实动作维度，保证随机动作宽度与 SDK 契约一致。
    backend.action_dim = env.action_dim

    class InProcessClient:
        """进程内策略客户端 — 绕过 HTTP 调用，直接生成动作。"""

        def infer(self, obs, instruction=None):
            """生成一批随机动作并做四元数安全处理。

            参数:
                obs: 观测对象；mock 后端只需要它支持 .to_dict()。
                instruction: 自然语言指令，透传给后端。

            返回:
                ActionChunk(actions=[ndarray(action_dim), ...])。

            说明: backend.infer 可能返回 (batch, dim) 数组或 list[ndarray]，
                  统一用 np.atleast_2d 归一到二维后逐条处理。
            """
            # backend.infer() 返回 numpy 数组或列表
            _raw = backend.infer([obs.to_dict()], instruction or "")
            _actions = [np.asarray(a, dtype=np.float32) for a in np.atleast_2d(_raw)]

            # ★ Patch 4/4: 四元数归一化 (零范数修复)
            # LEFT_WRIST_QUAT = 5:9, RIGHT_WRIST_QUAT = 12:16
            #
            # 为什么这两段是四元数: G1 WBC 的 23 维动作按
            #   [0:2] 导航, [2:5] 左 EEF 位置, [5:9] 左 EEF 姿态,
            #   [9:12] 右 EEF 位置, [12:16] 右 EEF 姿态, [16:23] 其余导航/躯干命令
            # 拼装，四元数只出现在两处姿态段，约定为 (w,x,y,z)。
            # 为什么零范数会报错: 单位四元数位于 S³，"全 0" 不在其中；
            #   下游 from_quat 需要先除以范数，scipy 会显式抛
            #   "Found zero norm quaternions in `quat`"。均匀随机采样命中
            #   零向量的概率极低但非零，一旦命中整条链路崩溃，故必须兜底。
            # 为什么补 [1,0,0,0]: (w,x,y,z)=(1,0,0,0) 是 wxyz 约定下的恒等旋转，
            #   语义上最保守；非零范数时归一化为合法单位四元数。
            for _a in _actions:
                for _s in [(5, 9), (12, 16)]:
                    _q = _a[_s[0]:_s[1]]
                    _norm = np.linalg.norm(_q)
                    if _norm < 1e-6:
                        _a[_s[0]:_s[1]] = [1.0, 0.0, 0.0, 0.0]  # 单位四元数
                    else:
                        _a[_s[0]:_s[1]] = _q / _norm
            return ActionChunk(actions=_actions)

    return InProcessClient()


# ═══════════════════════════════════════════════════════════════════
# 主程序 — 演示 LeRobot 兼容层的基本闭环
# ═══════════════════════════════════════════════════════════════════

def main():
    """
    主程序入口 — 演示 G1LocomanipPnPLeRobotEnv 的两种使用方式:
      方式 A: --success_test  → 验证 PnP 成功终止条件
      方式 B: --run_loop       → Mock 策略闭环比环

    参数: 无（全部来自 argparse）。
    返回: 无；结果通过日志打印（success_test 结果以 JSON 形式输出）。

    失败行为: 整体包在 try/except 中——异常被 logger.exception 记录后吞掉，
              finally 调用 env.close()。因此进程退出码恒为 0，排障需看日志。
      注意: env 的构造发生在 try 之外，若构造期间失败，Isaac Sim 不会被关闭。
    """
    # ── 命令行参数 ────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="ARENA 0.1.0 — G1 移动操作 PnP LeRobot 兼容环境"
    )
    # 与 pnp.py 相同: store_true + default=True 使 --headless 成为空操作，
    # 真正的开关是 --no_headless。
    parser.add_argument("--headless", action="store_true", default=True, help="无头模式")
    parser.add_argument("--no_headless", dest="headless", action="store_false", help="GUI 模式")
    parser.add_argument("--success_test", action="store_true", default=False,
                        help="★ 测试模式: teleport 箱子验证 PnP success 条件")
    parser.add_argument("--run_loop", action="store_true", default=False,
                        help="运行 Mock 策略基础闭环")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="闭环最大步数 (默认: 100)")
    parser.add_argument("--object", type=str, default="brown_box",
                        help="要抓取的物体")
    parser.add_argument("--embodiment", type=str, default="g1_wbc_pink",
                        help="机器人实施例")
    parser.add_argument("--backend", default="mock", choices=["mock", "http"],
                        help='策略后端: "mock"=随机动作 / "http"=VLA服务器')
    parser.add_argument("--server_url", default="http://127.0.0.1:8777/act",
                        help="VLA 策略服务器地址 (--backend http 时使用)")
    args = parser.parse_args()

    # ── 创建 LeRobot 兼容环境 ─────────────────────────────────────
    # enable_cameras 固定 False: Isaac Sim 5.1 的 camera data 0 dim 问题
    # 会让启用相机时的构建/步进崩溃。
    env = G1LocomanipPnPLeRobotEnv(
        headless=args.headless,
        enable_cameras=False,
        object_name=args.object,
        embodiment_name=args.embodiment,
        success_test=args.success_test,
    )

    try:
        if args.success_test:
            # ★ 方式 A: success_test 模式
            # 不做策略闭环，直接验证 PnP success 终止条件，返回结构化结果。
            _result = env.run_success_test()
            logger.info("✓ 实验结果: %s", json.dumps(_result, indent=2, ensure_ascii=False))

        elif args.run_loop:
            # ★ 方式 B: 策略闭环比环 (Mock 或 HTTP)
            _backend_type = args.backend
            logger.info("▶ 启动策略闭环比环 (max_steps=%d, backend=%s)...", args.max_steps, _backend_type)

            # 与 pnp.py 的 --instruction 默认值保持一致；注意本文件没有把
            # instruction 做成命令行参数，因此这里硬编码。
            _instruction = "pick up the brown box and place it in the blue bin"
            obs, info = env.reset()

            if _backend_type == "http":
                # ── HTTP 后端: 连接 VLA 服务器 ──────────────────────
                import json_numpy
                json_numpy.patch()  # numpy → JSON 序列化

                from arena.client import PolicyClient
                from arena.config import ClientConfig, AdapterConfig
                from arena.adapter import EmbodimentAdapter

                # 构建 HTTP 客户端 → POST /act
                _client = PolicyClient(ClientConfig(server_url=args.server_url))
                # 构建观测/动作适配器 (不归一化, 直接透传)
                _adapter = EmbodimentAdapter(
                    AdapterConfig(robot_type="unitree_g1", action_dim=env.action_dim),
                    instruction=_instruction,
                )
                logger.info("  → VLA 服务器: %s", args.server_url)

                # ★ HTTP 闭环: 观测 → 编码 → HTTP推理 → 解码 → env.step
                # 容错策略: 连续 3 次推理失败即终止循环，避免网络故障时刷屏；
                #          单次失败则降级为零动作(带合法四元数)继续跑。
                _error_count = 0
                _max_errors = 3
                for _step in range(args.max_steps):
                    # 从 obs 中取出关节位置作为 state；兼容扁平与 policy 嵌套两种组织。
                    _state = None
                    if isinstance(obs, dict):
                        _policy = obs.get("policy", {})
                        _state = _policy.get("robot_joint_pos")
                        if _state is None:
                            _state = obs.get("robot_joint_pos")

                    # 局部工具: 张量(可能带 grad) → 设备无关的 float32 一维 numpy。
                    def _to_np(v):
                        """把观测值规整成 float32 一维 numpy 数组。

                        参数:
                            v: torch.Tensor（可能在 GPU、可能带 grad）或任意数组式对象。
                        返回:
                            np.ndarray, dtype=float32, 已 flatten 的一维数组。
                        说明: 先 detach().cpu() 再转 numpy，否则 CUDA 张量会抛
                              "can't convert cuda:0 device type tensor to numpy"。
                        """
                        if isinstance(v, torch.Tensor):
                            v = v.detach().cpu()
                        return np.asarray(v, dtype=np.float32).reshape(-1)

                    # 图像是占位全黑图：相机被强制关闭，观测里没有真实 RGB。
                    _obs_dict = {
                        "images": {"head": np.zeros((224, 224, 3), dtype=np.uint8)},
                        "state": _to_np(_state) if _state is not None else np.zeros(0, dtype=np.float32),
                    }
                    _encoded = _adapter.encode_observation(_obs_dict)

                    # HTTP 推理, 容错降级为零动作
                    try:
                        _chunk = _client.infer(_encoded, _instruction)
                        _actions = _chunk.actions if hasattr(_chunk, 'actions') else _chunk
                        # 取 batch 中第一条动作（环境是单环境）。
                        _action = np.asarray(_actions[0] if len(_actions) > 0 else _actions, dtype=np.float32)
                        _error_count = 0
                    except Exception as _vla_err:
                        _error_count += 1
                        logger.warning("⚠ [step=%d] VLA error (连续 %d/%d): %s", _step, _error_count, _max_errors, str(_vla_err)[:120])
                        if _error_count >= _max_errors:
                            logger.error("✗ 连续 %d 次 VLA 错误, 终止", _max_errors)
                            break
                        # 降级动作: 全 0 会让四元数段零范数，故 w 分量置 1。
                        _action = np.zeros(env.action_dim, dtype=np.float32)
                        _action[5] = 1.0
                        _action[12] = 1.0

                    obs, reward, terminated, truncated, info = env.step(_action)
                    if terminated.any() or truncated.any():
                        # is_success 取自 step() 写入的 final_info（其值等于 terminated，
                        # 因此失败终止也会被打印为 success=True，属已知语义混淆）。
                        _success = info.get("final_info", {}).get("is_success", [False])
                        logger.info("✓ Episode 终止: step=%d success=%s", _step, _success[0])
                        break
                else:
                    # for-else: 循环正常走完（未 break）说明一直没终止。
                    logger.info("■ 达到 max_steps=%d 未终止", args.max_steps)

            else:
                # ── Mock 后端: 进程内随机动作 ───────────────────────
                _client = build_mock_client(env)

                for _step in range(args.max_steps):
                    # Fake Observation 对象 (Mock 客户端只需要 .to_dict())
                    # 用轻量适配器把 env 返回的 obs 包装成客户端期望的接口，
                    # 避免为了 mock 去构造完整的 ARENA Observation 对象。
                    class _FakeObs:
                        """最小观测适配器：mock 客户端只依赖 .to_dict() 一个方法。

                        定义在循环体内，因此每次迭代都会重新创建这个类（轻微开销，
                        属既有实现；见报告中的代码问题清单）。
                        """

                        # __init__: 暂存 env 返回的原始观测字典（单行写法保持不变）
                        def __init__(self, data): self._data = data
                        # to_dict: 按 mock 客户端期望的接口原样返回该字典（单行写法保持不变）
                        def to_dict(self): return self._data

                    _chunk = _client.infer(_FakeObs(obs))
                    _action = _chunk.actions[0] if hasattr(_chunk, 'actions') else _chunk

                    obs, reward, terminated, truncated, info = env.step(
                        np.asarray(_action, dtype=np.float32)
                    )

                    if terminated.any() or truncated.any():
                        _success = info.get("final_info", {}).get("is_success", [False])
                        logger.info("✓ Episode 终止: step=%d success=%s", _step, _success[0])
                        break
                else:
                    logger.info("■ 达到 max_steps=%d 未终止", args.max_steps)

        else:
            # 未给任何模式时只提示，不报错——保持入口"可安全空跑"。
            logger.info("■ 未指定操作。使用 --success_test 或 --run_loop。")

    except Exception as e:
        # 顶层兜底: 记录完整堆栈但不重新抛出，保证 finally 能关闭仿真。
        logger.exception("✗ 错误: %s", e)
    finally:
        env.close()
        logger.info("■ 仿真关闭完成。")


if __name__ == "__main__":
    main()