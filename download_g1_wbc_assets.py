#!/usr/bin/env python3
# 麻雀虽小智能科技（武汉）有限公司
"""下载 G1 WBC 全身控制机器人模型资产。

从 NVIDIA Omniverse Nucleus 服务器预下载到 ``/tmp/`` 本地缓存。

为什么需要这个脚本
------------------------------------------------------------------
Isaac Lab Arena 的 G1 移动操作实施例（``g1_wbc_pink``）默认从 NVIDIA 内部
开发服务器拉取资产::

    omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/

该地址在多数环境下**不可达**。两种应对方式：

1. 有权限时：用本脚本提前把资产抓到本地缓存（本脚本的用途）；
2. 无权限时：用 ``arena_g1_locomanip_pnp.py::_patch_g1_wbc_assets`` 把
   Nucleus URL 重定向到本地 GitHub 克隆的 Unitree URDF。

运行前提
------------------------------------------------------------------
必须在 **env_isaaclab** 环境里运行（需要 ``isaaclab`` 包），并且 Isaac Sim
要能正常启动 —— 因为 ``retrieve_file_path`` 依赖 Kit/Omniverse 运行时。

注意: 本脚本是**顺序执行**的顶层脚本（不是函数），导入即运行。
    ``AppLauncher`` 必须在导入 ``isaaclab.utils.assets`` 之前调用，
    否则 Kit 尚未初始化，资产检索会失败。
"""

from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 第 1 步：启动 Isaac Sim
# ---------------------------------------------------------------------------
# 必须最先执行：后面的 isaaclab.utils.assets 依赖 Kit 运行时。
# headless=True 表示不需要图形界面；enable_cameras=False 因为下载资产不需要相机。
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, enable_cameras=False)
sim_app = app_launcher.app
logger.info("✓ Isaac Sim 已启动")

# ---------------------------------------------------------------------------
# 第 2 步：下载 G1 WBC 资产
# ---------------------------------------------------------------------------
from isaaclab.utils.assets import retrieve_file_path

# 需要预取的三个条目：机器人模型目录、URDF 文件、以及 WBC 策略目录
# （内含 stand.onnx / walk.onnx 等下肢策略权重）
ASSETS = [
    "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/wbc_policy/robot_model/g1/",
    "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/wbc_policy/robot_model/g1/g1_29dof_with_hand.urdf",
    "omniverse://isaac-dev.ov.nvidia.com/Projects/nvblox/isaac_arena/g1_locomanip_assets/wbc_policy/",
]

for url in ASSETS:
    logger.info("▶ 下载: %s", url)
    try:
        # force_download=True 表示忽略已有缓存强制重新拉取
        local = retrieve_file_path(url, force_download=True)
        logger.info("  ✓ 已保存: %s", local)
    except Exception as e:
        # 单个条目失败不中断整体流程：某些资产可能确实无权限，
        # 但其余条目仍值得尝试下载。
        logger.warning("  ⚠ 下载失败: %s", e)

# ---------------------------------------------------------------------------
# 第 3 步：检查下载结果
# ---------------------------------------------------------------------------
# 遍历 /tmp 并按文件名关键词过滤，给出一份"实际落地了什么"的清单，
# 便于判断后续仿真能否正常加载资产。
logger.info("\n下载文件列表:")
for root, dirs, files in os.walk("/tmp"):
    for f in files:
        if "g1" in f.lower() or "wbc" in f.lower() or "robot_model" in f.lower():
            logger.info("  %s", os.path.join(root, f))

# 关闭 Isaac Sim；否则进程可能不退出
sim_app.close()
logger.info("■ 完成。")
