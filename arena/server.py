# 麻雀虽小智能科技（武汉）有限公司
"""VLA 策略服务器（FastAPI）。

技术报告对应第 2.3 / 2.4 节。

对外暴露所有客户端共用的两个端点::

    POST /act     {"observations": [Obs_dict, ...], "instruction": str}
               -> {"actions": [[...], ...], "latency_s": float}

    GET  /health  健康 -> 200, 不健康 -> 503

传输格式
------------------------------------------------------------------
图像与状态用 ``json_numpy`` 传输：它把 numpy 数组编码成带 ``"__numpy__"``
标记的字符串，使其能穿过普通 JSON 通道。为了照顾没装 ``json_numpy`` 的客户端，
也接受 ``{"encoded": "<json 字符串>"}`` 形式的**双重编码**载荷。

重要：``json_numpy.patch()`` 是**全局副作用**
------------------------------------------------------------------
``json_numpy.patch()`` 会直接替换 ``json.dumps`` / ``json.loads`` 两个模块级函数。
这意味着"是否 patch 过"会全局影响任何用到 ``json`` 的代码，包括：

* 客户端用 ``requests`` 发 numpy 载荷（``requests`` 内部走 ``json.dumps``）；
* 服务器把含 numpy 的返回体交给 FastAPI 序列化。

早期实现把这个 patch 放在 CLI 与集成脚本里，于是服务器能不能处理 numpy
**取决于调用方有没有先 patch** —— 属于典型的隐式依赖。现在改由服务器自己
在启动时与解码前调用 :func:`_ensure_json_numpy`，不再依赖外部。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict

import numpy as np

from arena.backends import PolicyBackend, build_backend
from arena.config import ServerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 版本与 json_numpy 补丁
# ---------------------------------------------------------------------------


def _arena_version() -> str:
    """读取 ARENA 包版本，失败时返回 ``"unknown"``。

    这里用函数内导入而不是顶层导入，避免 ``arena/__init__.py`` 与本模块
    在包初始化期间形成循环导入。
    """
    try:
        from arena import __version__

        return __version__
    except Exception:  # pragma: no cover - defensive
        return "unknown"


#: 进程级标记：确保 ``json_numpy.patch()`` 只执行一次。
_JSON_NUMPY_PATCHED = False


def _ensure_json_numpy() -> bool:
    """让 ``json`` 模块能够（反）序列化 numpy 数组，每进程只做一次。

    为什么必须在这里做
        patch 必须在**任何 numpy 载荷被编码之前**完成，否则标准库会抛
        ``TypeError: Object of type ndarray is not JSON serializable``。
        把这件事交给调用方会让服务器"碰巧能用"，因此改为服务器自身负责。

    Returns:
        True 表示 patch 可用；False 表示环境缺少 ``json_numpy``
        （此时只能走纯 JSON 路径，numpy 载荷会失败）。
    """
    global _JSON_NUMPY_PATCHED
    if _JSON_NUMPY_PATCHED:
        return True
    try:
        import json_numpy

        json_numpy.patch()
        _JSON_NUMPY_PATCHED = True
        logger.debug("json_numpy patch applied; numpy arrays are JSON-serializable")
        return True
    except ImportError:  # pragma: no cover - json_numpy 属于 server 可选依赖
        logger.warning(
            "json_numpy is unavailable: numpy arrays cannot be transported as JSON. "
            "Install with `pip install arena[server]` or send {'encoded': '<json>'}; "
            "note that plain `{'encoded': ...}` still needs json_numpy to carry ndarrays."
        )
        return False


def _decode_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """把收到的 JSON body 解码成规范 Python 对象（含 numpy 数组还原）。

    处理两种客户端形态：

    1. 直接发结构化 JSON（数组被 ``json_numpy`` 编码成 ``__numpy__`` 字符串）；
    2. 发 ``{"encoded": "<json>"}``（双重编码，兼容不支持 json_numpy 的客户端）。

    Args:
        payload: FastAPI 解析出的请求体字典。

    Returns:
        解码后的字典；若 ``json_numpy`` 缺失或解码失败，则退化为原始 payload
        （**不抛异常**，让上层决定是报错还是继续）。
    """
    if "encoded" in payload:
        payload = json.loads(payload["encoded"])
    if not _ensure_json_numpy():
        return payload
    try:
        import json_numpy

        # 绕一圈 dumps/loads 是为了把 __numpy__ 标记还原成真正的 ndarray
        return json_numpy.loads(json.dumps(payload))
    except Exception:  # noqa: BLE001 - payload 可能是不含 numpy 的纯 JSON
        return payload


# ---------------------------------------------------------------------------
# 服务器
# ---------------------------------------------------------------------------


class PolicyServer:
    """对 :class:`~arena.backends.PolicyBackend` 的薄封装，负责暴露 HTTP 路由。

    设计上刻意保持"薄"：不包含任何推理逻辑、不关心观测语义，
    只做"请求解码 → 调后端 → 响应编码"。
    """

    def __init__(self, backend: PolicyBackend) -> None:
        """绑定一个后端实例。

        Args:
            backend: 已构建好的后端（见 :func:`arena.backends.build_backend`）。
        """
        self.backend = backend

    def act(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """处理一次推理请求。

        容错的载荷解析：既支持标准的
        ``{"observations": [...], "instruction": str}``，也支持直接把单个观测
        当作整个 payload 传入。

        返回值刻意使用 **JSON 原生类型**（``actions`` 为嵌套 list）：
        旧实现返回 ``numpy.ndarray``，之所以能工作纯粹是因为
        ``json_numpy.patch()`` 在解码请求时顺手全局替换了 ``json.dumps``；
        任何未发送 numpy 载荷的调用方都会踩
        ``TypeError: Object of type ndarray is not JSON serializable``。

        Args:
            payload: 已解码的请求体。

        Returns:
            ``{"actions": [[float, ...], ...], "latency_s": float}``。
        """
        start = time.time()
        decoded = _decode_payload(payload)

        observations = decoded.get("observations")
        if observations is None:
            # 兼容：payload 本身就是单个观测
            observations = [decoded]
        if not isinstance(observations, list):
            observations = [observations]

        instruction = decoded.get("instruction")
        if instruction is None and observations:
            # 顶层没有指令时，回退取第一个观测里的 instruction
            instruction = observations[0].get("instruction", "")

        actions = np.asarray(self.backend.infer(observations, instruction or ""), dtype=np.float32)
        latency = time.time() - start
        logger.info("Inference complete in %.3fs (%d steps)", latency, len(actions))
        return {
            "actions": actions.tolist(),  # 转 JSON 原生类型（见 docstring）
            "latency_s": latency,
        }

    def health(self) -> Dict[str, Any]:
        """返回后端就绪状态，供 ``/health`` 路由使用。

        与"进程存活探针"不同，这里会**真正询问后端**：checkpoint 没加载成功、
        或上游不可达，都会得到 ``healthy: False``。

        **绝不抛异常** —— 如果 :meth:`PolicyBackend.status` 意外抛出，
        也会被捕获并转成 ``healthy: False`` + ``detail``，
        保证探活本身不会把服务器打挂。

        Returns:
            ``{"status", "healthy", "version", "backend", "detail", ...}``。
            ``healthy=False`` 时 ``status`` 为 ``"unavailable"``。
        """
        payload: Dict[str, Any] = {
            "status": "ok",
            "healthy": True,
            "version": _arena_version(),
        }
        try:
            payload.update(self.backend.status())
        except Exception as exc:  # noqa: BLE001 - 探活绝不允许抛出
            logger.exception("Health probe failed")
            payload.update(
                {"healthy": False, "backend": type(self.backend).__name__,
                 "detail": f"{type(exc).__name__}: {exc}"}
            )
        if not payload.get("healthy"):
            payload["status"] = "unavailable"
        return payload

    def run(self, host: str, port: int) -> None:  # pragma: no cover - 服务器主循环
        """启动 FastAPI / uvicorn 服务器（阻塞）。

        Args:
            host: 监听地址，``0.0.0.0`` 表示所有网卡。
            port: 监听端口，官方默认 8777。
        """
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse

        # 必须在开始服务之前 patch，否则携带 numpy 的请求会解码失败
        _ensure_json_numpy()

        app = FastAPI(
            title="ARENA VLA Policy Server",
            description="Vision-Language-Action inference API",
            version="0.1.0",
        )

        @app.get("/health")
        def health() -> JSONResponse:
            """真实探活：后端可服务返回 200，否则 503。

            返回 503 而非恒 200，是为了让负载均衡 / 编排器能直接按 HTTP
            状态码决定是否放流量。
            """
            payload = self.health()
            status_code = 200 if payload["healthy"] else 503
            return JSONResponse(payload, status_code=status_code)

        @app.post("/act")
        def act(payload: Dict[str, Any]) -> JSONResponse:
            """推理端点；内部异常统一转成 500 并带上异常类型便于定位。"""
            try:
                return JSONResponse(self.act(payload))
            except Exception as exc:  # pragma: no cover - 以 500 暴露给调用方
                logger.exception("Inference failed")
                return JSONResponse(
                    {"error": "inference_failed", "exception": type(exc).__name__},
                    status_code=500,
                )

        import uvicorn

        logger.info("Serving VLA policy on http://%s:%d/act", host, port)
        uvicorn.run(app, host=host, port=port, log_level="info")


def serve(config: ServerConfig) -> None:  # pragma: no cover - 进程入口
    """构建后端并启动 HTTP 服务器（阻塞）。

    Args:
        config: 服务器配置，``backend`` 决定用哪个后端实现。
    """
    backend = build_backend(config)
    PolicyServer(backend).run(config.host, config.port)
