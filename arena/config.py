# 麻雀虽小智能科技（武汉）有限公司
"""ARENA 配置对象与归一化工具。

本模块承担两件事：

1. **配置**：把"服务地址、模型路径、动作维度、归一化方式、域随机化等级"等
   参数从业务代码里抽出来，避免散落硬编码；
2. **归一化**：提供 ``normalize`` / ``unnormalize``，与 UnifoLM-VLA 训练管线保持一致。

为什么归一化必须与训练一致
------------------------------------------------------------------
VLA checkpoint 是在**归一化后的动作空间**上训练的：训练数据里的 action 与
proprio 会先按 ``dataset_statistics.json`` 的统计量映射到 ``[-1, 1]``，模型
学的就是这个空间。因此推理时：

* 送进模型的 proprio **必须**用同一套统计量归一化；
* 模型吐出的 action **必须**用同一套统计量反归一化，才能变回物理量。

用错统计量不会报错，只会让动作整体偏移/缩放 —— 属于最难排查的静默错误。
本项目用 :func:`load_norm_stats` 读 checkpoint 自带的
``dataset_statistics.json`` 来避免这个问题。

三种归一化方案
------------------------------------------------------------------
==============  ==========================================================
方案             公式
==============  ==========================================================
``NORMAL``      ``x_norm = (x - mean) / (std + 1e-8)``
``BOUNDS``      ``x_norm = 2*(x - low)/(high - low) - 1``，low/high 取 min/max
``BOUNDS_Q99``  同上，但 low/high 取 **q01/q99 分位数**
==============  ==========================================================

``BOUNDS_Q99`` 是默认值：用 1%/99% 分位数代替极值，可以避免个别离群点把
整个区间拉扁，鲁棒性更好。

掩码（mask）语义
------------------------------------------------------------------
统计量里带一个 ``mask`` 布尔向量：``mask=True`` 的维度做归一化，
``mask=False`` 的维度**原样透传**。常用来排除"无效/恒定"的动作维度
（例如某机器人没有被使用的关节）。归一化结果统一裁剪到 ``[-1, 1]``。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:  # PyYAML 在两个运行环境里都有，但仍做防御性导入
    import yaml
except ImportError:  # pragma: no cover - defensive
    yaml = None


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------


class NormalizationType(str, Enum):
    """支持的动作 / 本体感知归一化方案。

    继承 ``str`` 是为了让它可以被直接写进 YAML / JSON 而不需要额外转换。
    """

    NORMAL = "normal"          # 零均值、单位方差
    BOUNDS = "bounds"          # 按 min/max 映射到 [-1, 1]
    BOUNDS_Q99 = "bounds_q99"  # 按 q01/q99 分位数映射到 [-1, 1]


def _select_bounds(stats: Dict[str, Any], norm_type: NormalizationType):
    """取出指定归一化方案所需的 ``(low, high, mask)``。

    Args:
        stats: 单组统计量字典，例如 ``{"q01": [...], "q99": [...], "mask": [...]}``。
        norm_type: 归一化方案。``NORMAL`` 不是区间缩放，走到这里会抛错。

    Returns:
        ``(low, high, mask)`` 三个等长 ``float32``/``bool`` 数组。

    Raises:
        ValueError: 传入了未支持的归一化类型（例如误传 ``NORMAL``）。
    """
    # 维度先按 min 推断，没有 min 就按 q01 推断
    dim = len(stats.get("min", stats.get("q01", [])))
    if norm_type == NormalizationType.BOUNDS:
        mask = np.asarray(stats.get("mask", np.ones(dim, dtype=bool)), dtype=bool)
        low = np.asarray(stats["min"], dtype=np.float32)
        high = np.asarray(stats["max"], dtype=np.float32)
        return low, high, mask
    if norm_type == NormalizationType.BOUNDS_Q99:
        mask = np.asarray(stats.get("mask", np.ones(dim, dtype=bool)), dtype=bool)
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
        return low, high, mask
    raise ValueError(f"Unsupported normalization type: {norm_type}")


def normalize(values: np.ndarray, stats: Dict[str, Any], norm_type: NormalizationType) -> np.ndarray:
    """用 checkpoint 统计量把物理量归一化到模型空间。

    Args:
        values: 待归一化的数组（形状任意，按最后一维逐维处理）。
        stats: 单组统计量，含 ``mean/std`` 或 ``min/max`` 或 ``q01/q99``，可选 ``mask``。
        norm_type: 归一化方案。

    Returns:
        与输入同形状的 ``float32`` 数组。区间缩放的方案会被裁剪到 ``[-1, 1]``。
    """
    values = np.asarray(values, dtype=np.float32)
    if norm_type == NormalizationType.NORMAL:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        # +1e-8 防止除零（恒定维度 std=0 时很常见）
        return (values - mean) / (std + 1e-8)

    low, high, mask = _select_bounds(stats, norm_type)
    # 核心区间缩放公式：把 [low, high] 线性映射到 [-1, 1]
    normalized = 2.0 * (values - low) / (high - low + 1e-8) - 1.0
    # mask=False 的维度保持原值；最后统一裁剪，防止离群点溢出
    return np.clip(np.where(mask, normalized, values), -1.0, 1.0)


def unnormalize(normalized: np.ndarray, stats: Dict[str, Any], norm_type: NormalizationType) -> np.ndarray:
    """:func:`normalize` 的逆运算：把 ``[-1, 1]`` 还原成物理量。

    注意：区间缩放是**有损**的（裁剪到 [-1,1] 后信息已丢），因此
    ``unnormalize(normalize(x))`` 只在 x 原本落在区间内时严格相等。
    单元测试 ``test_normalization_round_trips`` 用的就是区间内的取值。

    Args:
        normalized: 模型空间的动作 / 状态。
        stats: 与归一化时相同的统计量。
        norm_type: 归一化方案。

    Returns:
        物理量数组，形状与输入一致。
    """
    normalized = np.asarray(normalized, dtype=np.float32)
    if norm_type == NormalizationType.NORMAL:
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return normalized * (std + 1e-8) + mean

    low, high, mask = _select_bounds(stats, norm_type)
    # 反解 x = (x_norm + 1) / 2 * (high - low) + low
    denormalized = 0.5 * (normalized + 1.0) * (high - low + 1e-8) + low
    return np.where(mask, denormalized, normalized)


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    """VLA 策略服务器配置。

    Attributes:
        backend: 后端类型，``mock`` | ``unifolm_vla`` | ``openpi`` | ``http``。
        host: 监听地址，``0.0.0.0`` 表示所有网卡（真机场景常需要）。
        port: 监听端口，默认 8777，与官方 UnifoLM-VLA 服务端一致。
        ckpt_path: ``unifolm_vla`` 后端的 checkpoint（``.pt``）路径。
            注意该文件所在目录的**上一级**必须同时存在 ``config.yaml`` 与
            ``dataset_statistics.json``，否则加载会失败，见
            ``unifolm-vla/src/unifolm_vla/model/framework/share_tools.py``。
        vlm_pretrained_path: 覆盖 checkpoint 里记录的 VLM 主干路径。
        unnorm_key: 反归一化所用的数据集键（多任务 checkpoint 需指定）。
        use_bf16: 是否用 bfloat16 推理，显存减半但需要较新 GPU。
        center_crop: 是否对图像做中心裁剪（与官方服务端预处理一致）。
        device: 推理设备，例如 ``cuda:0``；无 CUDA 时后端会自动回退 CPU。
        upstream_url: ``openpi`` / ``http`` 后端要转发到的上游 ``/act`` 地址。
    """

    backend: str = "mock"  # mock | unifolm_vla | openpi | http
    host: str = "0.0.0.0"
    port: int = 8777
    ckpt_path: str = ""
    vlm_pretrained_path: Optional[str] = None
    unnorm_key: str = "new_embodiment"
    use_bf16: bool = True
    center_crop: bool = True
    device: str = "cuda:0"
    upstream_url: Optional[str] = None


@dataclass
class ClientConfig:
    """策略客户端配置。

    Attributes:
        server_url: ``/act`` 端点完整地址。
        timeout_s: 单次请求超时（秒）。
        retries: 失败重试次数，``PolicyClient`` 内部按 0.05×attempt 退避。
        action_chunk_size: 期望的动作块长度，仅作记录/校验用途。
    """

    server_url: str = "http://127.0.0.1:8777/act"
    timeout_s: float = 10.0
    retries: int = 3
    action_chunk_size: int = 8


@dataclass
class AdapterConfig:
    """具身适配器配置（观测 + 动作双向转换）。

    ``action_dim`` / ``proprio_dim`` 只是**参考性占位值**，默认对齐 LIBERO
    checkpoint（7 / 8）。它们**不参与任何实际计算**：适配器的裁剪/填充一律
    以传入数组自身的形状为准（见 ``ActionAdapter.apply_joint_limits``）。

    各实施例的真实维度请显式传入，例如::

        LIBERO / 通用        :  7 / 8
        UnifoLM G1 (EE6D)    : 23 / 23   <- arena_g1_locomanip_pnp*.py
        GR1T2 微波炉          : 36 / -
        G1 关节空间控制        : 16 / 16

    对某个 UnifoLM-VLA checkpoint 而言，**权威维度由
    ``unifolm_vla.rlds_dataloader.constants`` 按平台决定**，不在本文件里。
    Agent 若看到此处与 constants 不一致，以 constants 为准。

    Attributes:
        robot_type: 机器人标识，仅用于日志/路由。
        image_size: 送入模型的正方形图像边长。
        action_dim: 参考动作维度（不作为计算依据）。
        proprio_dim: 参考本体感知维度（不作为计算依据）。
        normalization_type: 默认 ``bounds_q99``。
        joint_lower: 可选关节下限，用于动作裁剪；``None`` 表示不裁剪。
        joint_upper: 可选关节上限。
    """

    robot_type: str = "unitree_g1"
    image_size: int = 224
    action_dim: int = 7
    proprio_dim: int = 8
    normalization_type: NormalizationType = NormalizationType.BOUNDS_Q99
    joint_lower: Optional[List[float]] = None
    joint_upper: Optional[List[float]] = None


@dataclass
class Sim2RealConfig:
    """Sim-to-Real 域随机化课程配置。

    实际使用的扰动幅度来自 ``arena/sim2real.py::CURRICULUM[level]``，
    本数据类只负责"选等级 + 随机种子"。

    Attributes:
        level: 课程等级 1..4，会被裁剪到合法区间。
        observation_noise: 观测噪声（当前由 CURRICULUM 决定，此处保留兼容）。
        action_delay_steps: 动作延迟步数（同上）。
        image_augmentation: 是否启用图像增强（预留开关）。
        seed: 随机种子，保证扰动可复现。
    """

    level: int = 1  # 1..4（见 arena.sim2real）
    observation_noise: float = 0.0
    action_delay_steps: int = 0
    image_augmentation: bool = False
    seed: int = 42


@dataclass
class ArenaConfig:
    """ARENA 顶层配置，聚合四个子系统配置。

    支持 YAML 往返（:meth:`to_yaml` / :meth:`from_yaml`），便于把一次实验的
    全部参数固化下来，保证可复现。
    """

    server: ServerConfig = field(default_factory=ServerConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    sim2real: Sim2RealConfig = field(default_factory=Sim2RealConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转成纯 dict，并把枚举转成字符串以便序列化。

        ``dataclasses.asdict`` 会递归展开嵌套数据类，但不会处理 ``Enum``，
        因此这里单独把 ``normalization_type`` 换成它的字符串值。
        """
        data = asdict(self)
        data["adapter"]["normalization_type"] = self.adapter.normalization_type.value
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ArenaConfig":
        """从 dict 构造，缺失的字段用默认值补齐。

        Args:
            data: 可为 ``None``（等价于全默认配置）。各子配置键缺失时同样走默认值。

        Returns:
            构造好的 :class:`ArenaConfig`。
        """
        data = dict(data or {})
        adapter_data = dict(data.get("adapter", {}))
        if "normalization_type" in adapter_data:
            # 字符串 -> 枚举（YAML 读进来的是普通字符串）
            adapter_data["normalization_type"] = NormalizationType(
                adapter_data["normalization_type"]
            )
        return cls(
            server=ServerConfig(**data.get("server", {})),
            client=ClientConfig(**data.get("client", {})),
            adapter=AdapterConfig(**adapter_data),
            sim2real=Sim2RealConfig(**data.get("sim2real", {})),
        )

    @classmethod
    def from_yaml(cls, path) -> "ArenaConfig":
        """从 YAML 文件加载配置。

        Raises:
            RuntimeError: 环境里没有 PyYAML。
        """
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to load YAML configuration")
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_yaml(self, path) -> None:
        """把配置写成 YAML（字段顺序与数据类定义一致，便于人工 diff）。

        Raises:
            RuntimeError: 环境里没有 PyYAML。
        """
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to write YAML configuration")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def load_norm_stats(path) -> Dict[str, Any]:
    """加载 UnifoLM-VLA 的 ``dataset_statistics.json``。

    该文件由训练流程写出（``save_dataset_statistics``），结构大致为::

        {
          "<数据集/任务键>": {
            "action":  {"mean": [...], "std": [...], "min": [...], "max": [...],
                        "q01": [...], "q99": [...], "mask": [...]},
            "proprio": {...同结构...},
            "num_transitions": int,
            "num_trajectories": int
          },
          ...
        }

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的字典（按任务键分块）。
    """
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
