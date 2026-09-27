# 麻雀虽小智能科技（武汉）有限公司
"""ARENA 0.1.0 —— VLA Server–Client–Adapter 框架包入口。

本包实现技术报告描述的架构::

    VLA Policy Server (GPU)  <--HTTP-->  Client  <--->  Robot / Simulation
                                          |
                                    Embodiment Adapter

四个组成部分
------------------------------------------------------------------
* :mod:`arena.types`    —— 统一数据契约（``Observation`` / ``ActionChunk``）
* :mod:`arena.config`   —— 配置与归一化工具
* :mod:`arena.adapter`  —— 具身适配器（机器人空间 ↔ VLA 空间双向翻译）
* :mod:`arena.server`   —— 策略服务器（FastAPI + 可插拔后端）
* :mod:`arena.client`   —— 策略客户端 + 仿真/真机统一闭环
* :mod:`arena.sim2real` —— Sim-to-Real 域随机化课程（Level 1–4）
* :mod:`arena.cli`      —— ``arena server|sim|real`` 命令行入口

使用建议
------------------------------------------------------------------
只需要统一数据契约时::

    from arena import Observation, ActionChunk

需要跑完整闭环时，再进行深层导入以推迟重依赖加载::

    from arena.adapter import EmbodimentAdapter
    from arena.backends import build_backend

为什么这里只导出"轻量"对象
------------------------------------------------------------------
``Observation`` / ``ActionChunk`` / 各配置类都只依赖 numpy，顶层导出不会拖入
torch / fastapi / requests。这样 `import arena` 在最小环境下也能成功，
便于做环境自检（例如 ``python -c "import arena; print(arena.__version__)"``）。
"""

from arena.types import ActionChunk, Observation
from arena.config import (
    AdapterConfig,
    ArenaConfig,
    ClientConfig,
    NormalizationType,
    ServerConfig,
    Sim2RealConfig,
)

#: 当前版本号，与 ``pyproject.toml`` 保持一致；``/health`` 会回传该值。
__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Observation",
    "ActionChunk",
    "NormalizationType",
    "ArenaConfig",
    "ServerConfig",
    "ClientConfig",
    "AdapterConfig",
    "Sim2RealConfig",
]
