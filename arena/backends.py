# 麻雀虽小智能科技（武汉）有限公司
"""ARENA 策略服务器可插拔的 VLA 后端。

技术报告对应第 2.3 / 2.4 节。

后端契约
------------------------------------------------------------------
后端接收**已解码**的请求载荷::

    {"observations": [Obs_dict, ...], "instruction": str}

返回形状 ``(H, D)`` 的 ``float32`` 动作数组，且必须是**规范空间（未反归一化前）**
的动作 —— 反归一化由适配器负责，后端只对"模型自己的归一化空间"负责。

四个后端
------------------------------------------------------------------
================  ==========================================================
名称               说明
================  ==========================================================
``mock``          确定性后端，用于测试与干跑（无需 GPU / 网络）。
``unifolm_vla``   本地 UnifoLM-VLA checkpoint（需要 unifolm-vla 环境 + GPU）。
``openpi``        OpenPI 策略服务（HTTP）。
``http``          通用 HTTP 转发后端（透明代理）。
================  ==========================================================

设计要点
------------------------------------------------------------------
1. **可替换**：换模型只换后端名，服务器、客户端、适配器、控制回路都不动。
2. **探活是真实的**：:meth:`PolicyBackend.health` 不是恒返回 True 的占位，
   而是逐后端实探（见各实现与 :func:`probe_http_endpoint`）。
3. **重依赖惰性导入**：``torch`` / ``PIL`` / ``requests`` 都在方法内部导入，
   保证没有 GPU 的环境也能 ``import arena`` 并跑 mock 链路。
"""

from __future__ import annotations

import abc
import json
import logging
import zlib
from typing import Any, Dict, List, Tuple

import numpy as np

from arena.config import ServerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _stable_instruction_offset(instruction: str, slots: int = 100) -> float:
    """把指令字符串映射到一个小而**跨进程稳定**的偏移量。

    为什么不能用内置 ``hash()``
        CPython 对 ``str`` 默认启用哈希随机化（PEP 456 / PYTHONHASHSEED），
        同一个字符串在不同进程里 ``hash()`` 结果不同。若用它生成动作偏移，
        mock 后端每次运行都会给出不同动作，干跑与回归对比全部失效。
        实测：同一指令三次运行分别得到 0.080 / 0.033 / 0.036。

    为什么用 ``crc32``
        ``zlib.crc32`` 在跨进程、跨平台、跨 Python 版本下都稳定，且标准库自带、
        零依赖。默认 ``slots=100`` 时返回值落在 ``[0, 0.099]``，
        与修复前 ``(hash % 100)/1000`` 的取值范围完全一致。

    Args:
        instruction: 任务指令文本。
        slots: 离散档位数，决定偏移的取值粒度。

    Returns:
        ``[0, slots/10)`` 区间内的浮点偏移，例如 ``slots=100`` 时为 ``[0, 0.1)``。
    """
    return (zlib.crc32(instruction.encode("utf-8")) % slots) / float(slots * 10)


def probe_http_endpoint(url: str, timeout_s: float = 1.5) -> Tuple[bool, str]:
    """探测上游策略服务是否可服务，返回 ``(reachable, detail)``。

    **绝不抛异常** —— 它会被 ``/health`` 路由调用，任何异常都可能把服务器打挂，
    因此所有失败都转成 ``(False, 原因字符串)``。

    两级探测策略:

    1. ``GET <base>/health``：上游若实现了该路由（ARENA 服务器都实现），
       它的结论就是**权威结论，包括否定结论** —— 返回 5xx 表示上游自述不可用，
       转发后端绝不能把它美化成"健康"。
    2. ``HEAD <url>``：兜底，用于没有 health 路由的服务（例如官方
       UnifoLM-VLA 部署服务器）。此时任何 ``2xx``/``4xx`` 都算存活，
       因为兜底探测打的是 **POST-only 的 ``/act``**，它合法地回 ``405``。

    为什么兜底探测要拒绝 5xx
        代理 / 网关 / 容器网格对"无人监听的端口"常常直接回 ``502``。
        如果把 5xx 当存活，就会把"彻底不可用"报成"健康" —— 恰好掩盖了
        这个探活本来要暴露的故障。本机沙箱实测正是这种情况。

    Args:
        url: ``/act`` 端点地址（也接受 base 地址）。
        timeout_s: 单次请求超时（秒）。默认 1.5s，保证探活足够廉价。

    Returns:
        ``(是否可服务, 说明文本)``。说明文本会带上实际状态码，便于排障。
    """
    import requests  # 惰性导入：mock 链路无需 requests

    # 从 .../act 反推 base，以便拼出 .../health
    base = url.rsplit("/act", 1)[0] if url.endswith("/act") else url
    details: List[str] = []

    # ---- 第一级：GET /health（权威） ----
    health_url = base.rstrip("/") + "/health"
    try:
        response = requests.get(health_url, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - 探活绝不允许抛出
        details.append(f"GET {health_url}: {type(exc).__name__}")
    else:
        line = f"GET {health_url} -> {response.status_code}"
        if 200 <= response.status_code < 300:
            # 2xx：尝试读取响应体里的 healthy 字段（ARENA /health 会带）
            try:
                body = response.json()
            except ValueError:
                return True, f"reachable ({line})"  # 非 JSON，仅按可达处理
            if isinstance(body, dict) and "healthy" in body:
                return bool(body["healthy"]), f"{line} healthy={body['healthy']}"
            return True, f"reachable ({line})"
        if 500 <= response.status_code < 600:
            # 上游答复了，而答复内容是"我不可用" -> 直接判不健康
            return False, f"upstream unhealthy ({line})"
        details.append(line)  # 4xx：说明没有 health 路由，转兜底探测

    # ---- 第二级：HEAD /act（兜底） ----
    try:
        response = requests.head(url, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - 探活绝不允许抛出
        details.append(f"HEAD {url}: {type(exc).__name__}")
    else:
        line = f"HEAD {url} -> {response.status_code}"
        if 200 <= response.status_code < 500:
            # 4xx（尤其 405 Method Not Allowed）同样证明服务在监听
            return True, f"reachable ({line})"
        details.append(line)

    return False, "; ".join(details) or "unreachable"


# ---------------------------------------------------------------------------
# 抽象基类
# ---------------------------------------------------------------------------


class PolicyBackend(abc.ABC):
    """所有策略后端的抽象基类。

    子类必须实现 :meth:`infer`；建议按需覆写 :meth:`health` 与 :meth:`status`。
    """

    def __init__(self, config: ServerConfig) -> None:
        """保存服务器配置（含模型路径、设备、上游地址等）。"""
        self.config = config

    @abc.abstractmethod
    def infer(self, observations: List[Dict[str, Any]], instruction: str) -> np.ndarray:
        """执行一次策略推理。

        Args:
            observations: 观测字典列表（每个为 ``Observation.to_dict()`` 的产物）。
                支持批量，长度通常为 1。
            instruction: 自然语言任务描述。

        Returns:
            形状 ``(H, action_dim)`` 的 ``float32`` 动作块。
        """

    def health(self) -> bool:
        """报告后端当前能否服务推理。

        与 :meth:`infer` 不同，本方法**不得抛异常**，并且要足够廉价以便被轮询。
        依赖远端服务或已加载 checkpoint 的子类应当覆写它。

        Returns:
            默认 True（适用于无外部依赖的后端）。
        """
        return True

    def status(self) -> Dict[str, Any]:
        """返回供 ``/health`` 路由使用的结构化状态。

        统一在此处兜住 :meth:`health` 可能抛出的异常，保证"探活失败"表现为
        ``healthy=False`` 而不是 500。子类通常调用 ``super().status()``
        后再往返回字典里补充自己的字段。

        Returns:
            ``{"backend": 类名, "healthy": bool, "detail": str | None}``。
        """
        try:
            healthy = bool(self.health())
            detail: Any = None
        except Exception as exc:  # noqa: BLE001 - 探活绝不允许抛出
            healthy, detail = False, f"{type(exc).__name__}: {exc}"
        return {
            "backend": type(self).__name__,
            "healthy": healthy,
            "detail": detail,
        }

    def close(self) -> None:  # pragma: no cover - 可选钩子
        """释放后端持有的资源（可选实现）。"""


# ---------------------------------------------------------------------------
# 后端 1：mock（确定性，无需 GPU）
# ---------------------------------------------------------------------------


class MockPolicyBackend(PolicyBackend):
    """确定性后端：产生平滑、有界的动作，用于端到端冒烟测试。

    动作由"观测状态的 tanh + 从 0 到 1 的斜坡 + 指令相关的固定偏移"构成，
    因此**可复现、有界、且随指令变化**。它不学习任何东西，只用来验证
    "适配器 → 服务器 → 客户端 → 动作"整条管线是否接通。

    注意: ``action_dim`` 默认 7（LIBERO 量级），但集成脚本会在构造后直接改写
        ``backend.action_dim = <env 实际维度>``。若维度不匹配，多余维度会被
        静默补零/截断，因此**务必设置成目标环境的真实动作维度**。
    """

    def __init__(self, config: ServerConfig, action_dim: int = 7, horizon: int = 8) -> None:
        """初始化。

        Args:
            config: 服务器配置（mock 不使用其中字段）。
            action_dim: 动作维度，默认 7。
            horizon: 动作块长度 H，默认 8。
        """
        super().__init__(config)
        self.action_dim = action_dim
        self.horizon = horizon

    def infer(self, observations: List[Dict[str, Any]], instruction: str) -> np.ndarray:
        """根据观测状态与指令生成确定性的 ``(H, action_dim)`` 动作块。

        生成规则::

            state  = 观测 state 的前 action_dim 维（不足补零）
            offset = _stable_instruction_offset(instruction)   # 每个任务一个固定偏移
            ramp   = linspace(0, 1, H)                        # 沿时间轴从 0 渐增到 1
            action = 0.05 * ramp * tanh(state) + offset

        取 ``tanh`` 是为了把状态压到 ``(-1, 1)``，避免大状态值放大成越界动作。

        Args:
            observations: 观测列表；只取**最后一个**观测的 ``state`` 作为当前状态。
            instruction: 指令文本，决定固定偏移量。

        Returns:
            ``(horizon, action_dim)`` ``float32`` 数组。
        """
        state = np.zeros(self.action_dim, dtype=np.float32)
        if observations:
            last_state = observations[-1].get("state")
            if last_state is not None:
                flat = np.asarray(last_state, dtype=np.float32).reshape(-1)
                width = min(self.action_dim, flat.shape[0])
                state[:width] = flat[:width]  # 只填公共前缀，多余维度保持 0

        # 指令校验和给出每个任务一个稳定且互不相同的偏移
        offset = _stable_instruction_offset(instruction)
        ramp = np.linspace(0.0, 1.0, self.horizon, dtype=np.float32)[:, None]
        base = np.tanh(state)[None, :]
        actions = 0.05 * ramp * base + offset
        return actions.astype(np.float32)


# ---------------------------------------------------------------------------
# 后端 2：本地 UnifoLM-VLA checkpoint
# ---------------------------------------------------------------------------


class UnifoLMVLABackend(PolicyBackend):
    """本地 UnifoLM-VLA checkpoint 后端。

    与官方 ``deployment/model_server/run_real_eval_server.py`` 等价：
    构造 Qwen 对话模板 → 归一化 proprio → 运行 DiT 动作头 → 反归一化动作块。

    两条重要约束:
        1. ``ckpt_path`` 指向的 ``.pt`` 文件**所在目录的上一级**必须同时存在
           ``config.json``/``config.yaml`` 与 ``dataset_statistics.json``，
           否则 ``from_pretrained`` 会失败（见 share_tools.read_mode_config）。
        2. 归一化类型被硬编码为 ``BOUNDS_Q99``（与 G1 系列常量一致）。
           若换成使用 ``BOUNDS`` 的 checkpoint（如 G1 joint 控制），
           这里需要改为读取 ``constants.ACTION_PROPRIO_NORMALIZATION_TYPE``。
    """

    def __init__(self, config: ServerConfig) -> None:
        """加载 checkpoint 到目标设备。

        Args:
            config: 服务器配置；``ckpt_path`` 必填，``device`` 指定优先设备。

        Raises:
            FileNotFoundError: checkpoint 或其配套 JSON 不存在。
            RuntimeError: 权重与模型结构不匹配。
        """
        super().__init__(config)
        import torch  # 惰性导入，保证无 torch 环境可用 mock 链路

        from unifolm_vla.model.framework.base_framework import baseframework

        self.torch = torch
        # 没有 CUDA 时静默回退 CPU：保证可在纯 CPU 机器上做接口联调
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        logger.info("Loading UnifoLM-VLA checkpoint from %s", config.ckpt_path)
        vla = baseframework.from_pretrained(
            config.ckpt_path, vlm_pretrained_path=config.vlm_pretrained_path
        )
        if config.use_bf16:
            vla = vla.to(torch.bfloat16)  # 推理显存减半
        self.vla = vla.to(self.device).eval()
        self.processor = self.vla.qwen_vl_interface.processor
        self.norm_stats = self.vla.norm_stats  # 按任务键分块的统计量
        logger.info("UnifoLM-VLA model ready on %s", self.device)

    def _stats_for(self, task_name):
        """取出某任务对应的统计量、归一化类型与归一化函数。

        Args:
            task_name: 任务/数据集键；``None`` 时回退 ``config.unnorm_key``。

        Returns:
            ``(stats, norm_type, normalize, unnormalize)`` 四元组，
            其中 ``stats`` 含 ``"action"`` 与 ``"proprio"`` 两个子块。
        """
        from arena.config import NormalizationType, normalize, unnormalize

        key = task_name or self.config.unnorm_key
        stats = self.norm_stats[key]
        norm_type = NormalizationType.BOUNDS_Q99
        return stats, norm_type, normalize, unnormalize

    def health(self) -> bool:
        """checkpoint 已加载到设备即视为就绪。

        加载动作在 ``__init__`` 中完成且失败会抛异常，因此能走到这里就说明
        权重、processor、归一化统计量都已具备。

        注意: 配置了 CUDA 却静默回退到 CPU 时，本方法**仍返回 True**（接口确实
            可用），但会在 :meth:`status` 的 ``detail`` 里明确标注，
            以免"以为在用 GPU"这种隐性降级无人察觉。
        """
        return getattr(self, "vla", None) is not None

    def status(self) -> Dict[str, Any]:
        """在基类状态上补充设备、CUDA 可用性、任务键与统计量键列表。"""
        payload = super().status()
        device = str(getattr(self, "device", "unknown"))
        payload["device"] = device
        payload["cuda_available"] = bool(
            getattr(getattr(self, "torch", None), "cuda", None) is not None
            and self.torch.cuda.is_available()
        )
        payload["unnorm_key"] = self.config.unnorm_key
        payload["norm_stat_keys"] = sorted(getattr(self, "norm_stats", {}) or {})
        if device == "cpu" and str(self.config.device).startswith("cuda"):
            payload["detail"] = (
                f"requested {self.config.device} but CUDA is unavailable; running on CPU"
            )
        return payload

    def infer(self, observations: List[Dict[str, Any]], instruction: str) -> np.ndarray:
        """执行一次 VLA 推理。

        流程与官方服务端一致：

        1. 反序列化观测 → 逐观测收集全部图像（第三人称在前、腕部在后）；
        2. 缩放图像到 224×224 并转 PIL；
        3. 按 Qwen 对话模板拼 ``The task is "<instruction>".`` 文本；
        4. proprio 归一化后作为 ``state`` 输入；
        5. 调用 ``predict_action`` 得到**归一化空间**的动作；
        6. 反归一化回物理量返回。

        Args:
            observations: 观测字典列表。
            instruction: 任务指令（内部转小写，与训练时一致）。

        Returns:
            反归一化后的 ``(H, action_dim)`` 动作块。

        Note:
            ``proprio`` 用 ``np.stack`` 拼批，但统计量取自**第一个观测**的
            ``task_name``。若同批混用不同任务，会得到不一致的归一化基准。
        """
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        from arena.adapter import resize_image
        from arena.types import Observation

        parsed = [Observation.from_dict(obs) for obs in observations]
        task_name = parsed[0].task_name if parsed else None
        stats, norm_type, normalize, unnormalize = self._stats_for(task_name)

        # 收集图像：与官方服务端相同的顺序（先 full，再各 wrist）
        images: List[Image.Image] = []
        for obs in parsed:
            for image in obs.all_images:
                images.append(Image.fromarray(resize_image(image, 224)).convert("RGB"))

        text = f'The task is "{instruction.lower()}".'
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": text},
                ],
            }
        ]
        # 走 Qwen processor 的对话模板 + 视觉信息提取
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(
            text=chat,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # proprio 归一化 -> GPU 张量；unsqueeze(0) 补上 batch 维
        proprio = np.stack([obs.state for obs in parsed], axis=0)
        normalized_state = normalize(proprio, stats["proprio"], norm_type)
        batch_input["state"] = (
            torch.from_numpy(normalized_state).unsqueeze(0).to(self.device)
        )
        for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
            if key in batch_input:
                batch_input[key] = batch_input[key].to(self.device)

        with torch.inference_mode():  # 关闭梯度，省显存
            result = self.vla.predict_action(qwen_inputs=batch_input)
        normalized_actions = result["normalized_actions"][0]
        return unnormalize(normalized_actions, stats["action"], norm_type)


# ---------------------------------------------------------------------------
# 后端 3：OpenPI（HTTP）
# ---------------------------------------------------------------------------


class OpenPIBackend(PolicyBackend):
    """OpenPI ``serve_policy.py`` 策略服务的 HTTP 客户端。

    与 :class:`HTTPForwardBackend` 的区别：本后端按 OpenPI 的**明文 JSON** 协议
    发送 ``{"observations": ..., "instruction": ...}``，不做 ``json_numpy`` 双重编码。
    """

    def __init__(self, config: ServerConfig) -> None:
        """确定上游地址。

        Args:
            config: ``upstream_url`` 优先；未提供时按
                ``http://127.0.0.1:{port}/act`` 推断。
        """
        super().__init__(config)
        self.url = config.upstream_url or f"http://127.0.0.1:{config.port}/act"

    def health(self) -> bool:
        """探测上游 OpenPI/ARENA 端点是否可服务（不抛异常）。"""
        reachable, detail = probe_http_endpoint(self.url)
        self._last_probe_detail = detail  # 供 status() 复用，避免重复探测
        return reachable

    def status(self) -> Dict[str, Any]:
        """在基类状态上补充上游地址与探测详情。"""
        payload = super().status()
        payload["upstream_url"] = self.url
        if payload.get("detail") is None:
            payload["detail"] = getattr(self, "_last_probe_detail", None)
        return payload

    def infer(self, observations: List[Dict[str, Any]], instruction: str) -> np.ndarray:
        """POST 到上游并解析动作。

        兼容两种响应：``{"actions": [...]}`` 或直接返回动作数组。

        Args:
            observations: 观测字典列表。
            instruction: 任务指令。

        Returns:
            ``(H, D)`` ``float32`` 动作数组。

        Raises:
            requests.HTTPError: 上游返回非 2xx。
        """
        import requests

        payload = {"observations": observations, "instruction": instruction}
        response = requests.post(self.url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "actions" in data:
            return np.asarray(data["actions"], dtype=np.float32)
        return np.asarray(data, dtype=np.float32)


# ---------------------------------------------------------------------------
# 后端 4：透明 HTTP 转发
# ---------------------------------------------------------------------------


class HTTPForwardBackend(PolicyBackend):
    """把请求透明转发到另一个 ARENA/OpenPI 服务器。

    与 :class:`OpenPIBackend` 的差别：本后端使用 ``{"encoded": "<json>"}``
    **双重编码**格式，使 numpy 数组可以穿过不支持 ``json_numpy`` 的客户端；
    并且能识别上游返回的 ``__numpy__`` 编码结果。
    """

    def __init__(self, config: ServerConfig) -> None:
        """初始化。

        Args:
            config: 必须提供 ``upstream_url``。

        Raises:
            ValueError: 未提供 ``upstream_url``（转发目标不可推断）。
        """
        super().__init__(config)
        if not config.upstream_url:
            raise ValueError("HTTPForwardBackend requires `upstream_url`")
        self.url = config.upstream_url

    def health(self) -> bool:
        """探测上游是否可服务。

        转发者只能和它转发的服务一样健康：上游不可达就报不健康，
        而不是因为"本进程还活着"就报健康。
        """
        reachable, detail = probe_http_endpoint(self.url)
        self._last_probe_detail = detail
        return reachable

    def status(self) -> Dict[str, Any]:
        """在基类状态上补充上游地址与探测详情。"""
        payload = super().status()
        payload["upstream_url"] = self.url
        if payload.get("detail") is None:
            payload["detail"] = getattr(self, "_last_probe_detail", None)
        return payload

    def infer(self, observations: List[Dict[str, Any]], instruction: str) -> np.ndarray:
        """转发推理请求并解析动作。

        解析顺序：``{"actions": ...}`` → ``{"__numpy__": ...}`` → 直接当数组。

        Args:
            observations: 观测字典列表。
            instruction: 任务指令。

        Returns:
            ``(H, D)`` ``float32`` 动作数组。
        """
        import requests

        payload = {"encoded": json.dumps({"observations": observations, "instruction": instruction})}
        response = requests.post(self.url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            if "actions" in data:
                return np.asarray(data["actions"], dtype=np.float32)
            if "__numpy__" in data:
                # 上游用 json_numpy 编码了整个动作数组
                try:
                    import json_numpy

                    return np.asarray(json_numpy.loads(json.dumps(data)), dtype=np.float32)
                except Exception:  # pragma: no cover - 尽力而为，失败则走下面的兜底
                    pass
        return np.asarray(data, dtype=np.float32)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

#: 后端名 → 实现类的注册表。新增后端只需在此登记。
_BACKENDS = {
    "mock": MockPolicyBackend,
    "unifolm_vla": UnifoLMVLABackend,
    "openpi": OpenPIBackend,
    "http": HTTPForwardBackend,
}


def build_backend(config: ServerConfig) -> PolicyBackend:
    """按 ``config.backend`` 实例化对应后端。

    Args:
        config: 服务器配置，``backend`` 字段决定实现类。

    Returns:
        构建好的后端实例。

    Raises:
        ValueError: 后端名未注册（错误信息会列出所有可用名称）。

    示例:
        >>> backend = build_backend(ServerConfig(backend="mock"))
        >>> backend.health()
        True
    """
    if config.backend not in _BACKENDS:
        raise ValueError(
            f"Unknown backend '{config.backend}'. "
            f"Available: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[config.backend](config)
