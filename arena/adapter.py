# 麻雀虽小智能科技（武汉）有限公司
"""具身适配器（Embodiment Adapter）：机器人原生 I/O ↔ 规范 VLA 空间的双向翻译。

技术报告对应第 4 节。核心思想是**把所有"机器人差异"集中到一个薄层里**，
让"模型空间"与"机器人空间"彻底解耦::

    Obs_robot    --encode_observation--> Obs_VLA        (上行)
    Action_VLA   --decode_action-------> Action_robot   (下行)

为什么需要适配器（而不是让模型直接吃机器人数据）
------------------------------------------------------------------
真实困难不在于"连上网"，而在于：不同机器人有不同数量的关节、相机内外参、
坐标系约定、控制频率与动作定义。例如：

* 通用配置可能只有较小的标准动作维度（如 LIBERO 的 7 维）；
* UnifoLM-VLA / G1 具体链路可能使用 23 维末端位姿动作；
* G1 WBC 底层实例还可能扩展到更多关节/控制变量。

如果让 VLA 主干去感知这些差异，那么每换一台机器人就要动模型代码。因此
本项目坚持一条原则：**模型只认统一数据；机器人差异不进入 VLA 核心代码。**

本模块拆成三层职责
------------------------------------------------------------------
* :class:`ObservationAdapter` —— 缩放/格式化相机画面、归一化 proprio、
  组装规范观测；
* :class:`ActionAdapter` —— 反归一化预测动作、按关节限位裁剪；
* :class:`EmbodimentAdapter` —— 组合上述两者，是控制回路唯一需要交互的对象。

注意"动作维度"的处理方式
------------------------------------------------------------------
适配器**不假设**固定的 action_dim。所有裁剪/填充都按传入数组的实际形状
推导（见 :meth:`ActionAdapter.apply_joint_limits` 取三者最小值），
因此同一个适配器实例可以处理 7 维、23 维或 36 维动作，只要归一化统计量匹配。

图像缩放为什么用纯 NumPy 实现
------------------------------------------------------------------
``resize_image`` 用双线性插值手写，不引入 OpenCV / torchvision。原因是本项目
需要能在"只有 numpy"的最小环境里导入（``demo.py``、单元测试、CI 都要跑），
同时保持与 VLA 服务端一致的 ``(size, size, 3)`` 契约。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from arena.config import AdapterConfig, NormalizationType, normalize, unnormalize
from arena.types import ActionChunk, Observation, VLA_IMAGE_SIZE


# ---------------------------------------------------------------------------
# 图像预处理工具（模块级函数，独立可测）
# ---------------------------------------------------------------------------


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """把任意相机帧规整为 ``(H, W, 3)`` 的 ``uint8`` RGB 数组。

    需要处理的常见输入差异：

    * 灰度图 ``(H, W)`` → 复制三通道；
    * RGBA ``(H, W, 4)`` → 丢弃 alpha 通道；
    * 浮点图 ``[0, 1]`` → 乘 255；浮点图 ``[0, 255]`` → 仅裁剪；
    * 其它整型（如 ``int32``）→ 直接转 ``uint8``。

    最后调用 ``ascontiguousarray`` 保证内存连续，避免后续切片时出现
    非连续数组导致的隐式拷贝或 reshape 报错。

    Args:
        image: 任意相机帧。

    Returns:
        ``(H, W, 3)`` ``uint8`` 且内存连续的 RGB 数组。
    """
    image = np.asarray(image)
    if image.ndim == 2:  # 灰度 -> RGB
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:  # RGBA -> RGB
        image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            # 用最大值判断量纲：[0,1] 需要放大，[0,255] 只需裁剪
            scale = 255.0 if image.max() <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255)
        image = image.astype(np.uint8)
    return np.ascontiguousarray(image)


def resize_image(image: np.ndarray, size: int = VLA_IMAGE_SIZE) -> np.ndarray:
    """用纯 NumPy 双线性插值把图像缩放到 ``(size, size, 3)``。

    之所以不依赖 OpenCV：保持适配器在任意环境可导入，同时满足 VLA 服务端
    对输入尺寸的硬性契约。

    算法（可分离双线性插值）:
        1. 在源图上按 ``linspace(0, src-1, size)`` 取目标采样坐标
           —— 使用端点对齐（align_corners=True 风格），与训练数据预处理一致；
        2. 分别取左右/上下邻居索引与权重；
        3. 先在 x 方向插值得到 top/bottom，再在 y 方向插值。
        x、y 分离可以把 ``O(size²×4)`` 的运算拆成两次一维运算。

    Args:
        image: 任意相机帧（会先经 :func:`_to_uint8_rgb` 规整）。
        size: 目标正方形边长，默认 :data:`arena.types.VLA_IMAGE_SIZE`（224）。

    Returns:
        ``(size, size, 3)`` ``uint8`` 数组。若输入已是目标尺寸则原样返回。
    """
    image = _to_uint8_rgb(image)
    if image.shape[0] == size and image.shape[1] == size:
        return image  # 已是目标尺寸，避免无谓重采样造成画质损失

    src_h, src_w = image.shape[:2]
    # 目标像素在源图上的浮点坐标（端点对齐）
    dst_indices_y = np.linspace(0, src_h - 1, size)
    dst_indices_x = np.linspace(0, src_w - 1, size)

    # 四个邻居索引；+1 越界时裁剪到边界（等价于 edge padding）
    y0 = np.floor(dst_indices_y).astype(np.int64)
    x0 = np.floor(dst_indices_x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)

    # 插值权重，形状做成可广播：y 方向 (size,1,1)，x 方向 (1,size,1)
    wy = (dst_indices_y - y0)[:, None, None]
    wx = (dst_indices_x - x0)[None, :, None]

    # 先沿 x 插值，再沿 y 插值（可分离 => 更快且实现更短）
    top = image[y0][:, x0] * (1 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1 - wx) + image[y1][:, x1] * wx
    resized = top * (1 - wy) + bottom * wy
    return np.clip(resized, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 观测适配器
# ---------------------------------------------------------------------------


class ObservationAdapter:
    """把机器人观测转换为规范 VLA 观测（上行方向）。

    职责边界：只做**格式与量纲**转换（缩放图像、归一化状态、统一键名），
    不做任何语义判断（例如"该用哪台相机"由上层决定）。
    """

    def __init__(
        self,
        config: Optional[AdapterConfig] = None,
        norm_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """初始化。

        Args:
            config: 适配器配置；``None`` 时使用默认 :class:`AdapterConfig`。
            norm_stats: checkpoint 的归一化统计量。为空时 :meth:`encode_state`
                **不做归一化**（原样透传），便于无 checkpoint 的冒烟测试。
        """
        self.config = config or AdapterConfig()
        self.norm_stats = norm_stats or {}

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """缩放并规整单张相机画面。

        Args:
            image: 任意相机帧。

        Returns:
            ``(image_size, image_size, 3)`` ``uint8`` 数组。
        """
        return resize_image(_to_uint8_rgb(image), self.config.image_size)

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """按 checkpoint 统计量归一化本体感知状态。

        容错设计：统计量缺失时**不报错、原样返回**。这样在只有 mock 后端的
        干跑场景下链路依然可用。

        Args:
            state: 原始 proprio 向量（会被展平为一维 ``float32``）。

        Returns:
            归一化后的状态；无统计量时为原值。
        """
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if not self.norm_stats:
            return state
        # 优先取 "proprio" 子块；兼容"stats 本身就是 proprio 统计量"的旧格式
        proprio_stats = self.norm_stats.get("proprio", self.norm_stats)
        if not proprio_stats:
            return state
        return normalize(state, proprio_stats, self.config.normalization_type)

    def encode_observation(
        self,
        images: Dict[str, np.ndarray],
        state: np.ndarray,
        instruction: str,
        task_name: Optional[str] = None,
    ) -> Observation:
        """组装规范 :class:`~arena.types.Observation`。

        Args:
            images: 相机名 → 画面，例如 ``{"head": ..., "wrist": ...}``。
                键名会被**统一转小写**（``"Head"`` → ``"head"``），
                以免因大小写不一致导致下游找不到相机。含 ``wrist`` 的键
                会被视为眼在手上相机。
            state: 原始 proprio 向量。
            instruction: 自然语言任务描述。
            task_name: 可选数据集键，用于选择对应的归一化统计量。

        Returns:
            组装好的规范观测对象。
        """
        encoded_images: Dict[str, np.ndarray] = {}
        for key, image in images.items():
            encoded_images[key.lower()] = self.encode_image(image)
        return Observation(
            images=encoded_images,
            state=self.encode_state(state),
            instruction=instruction,
            task_name=task_name,
        )

    def encode(self, robot_observation: Dict[str, Any]) -> Observation:
        """:meth:`encode_observation` 的字典入参便捷封装。

        Args:
            robot_observation: 至少包含 ``images`` 与 ``state`` 键；
                ``instruction`` / ``task_name`` 可选。

        Returns:
            组装好的规范观测对象。
        """
        return self.encode_observation(
            images=robot_observation["images"],
            state=robot_observation["state"],
            instruction=robot_observation.get("instruction", ""),
            task_name=robot_observation.get("task_name"),
        )


# ---------------------------------------------------------------------------
# 动作适配器
# ---------------------------------------------------------------------------


class ActionAdapter:
    """把规范 VLA 动作转换为机器人原生命令（下行方向）。

    处理顺序固定为：**反归一化 → 关节限位裁剪**。
    顺序不可颠倒：先裁剪会把模型空间的合理值误伤，导致动作被静默削弱。
    """

    def __init__(
        self,
        config: Optional[AdapterConfig] = None,
        norm_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        """初始化。

        Args:
            config: 适配器配置，``joint_lower``/``joint_upper`` 决定是否裁剪。
            norm_stats: checkpoint 归一化统计量；为空时不做反归一化。
        """
        self.config = config or AdapterConfig()
        self.norm_stats = norm_stats or {}

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        """反归一化并裁剪**单个**动作向量。

        Args:
            action: 模型输出的单步动作（会被展平为一维）。

        Returns:
            机器人可直接执行的物理量动作。
        """
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.norm_stats:
            # 优先取 "action" 子块；兼容旧格式
            action_stats = self.norm_stats.get("action", self.norm_stats)
            if action_stats:
                action = unnormalize(action, action_stats, self.config.normalization_type)
        return self.apply_joint_limits(action)

    def decode_chunk(self, action_chunk: np.ndarray) -> ActionChunk:
        """逐行解码 ``(H, D)`` 动作块。

        逐行而不是整体矩阵运算，是因为 :meth:`apply_joint_limits` 需要对
        每个向量单独裁剪；H 通常很小（8~25），性能可忽略。

        Args:
            action_chunk: ``(H, D)``，或可广播成二维的数组。

        Returns:
            解码后的 :class:`~arena.types.ActionChunk`。
        """
        actions = np.atleast_2d(np.asarray(action_chunk, dtype=np.float32))
        decoded = np.stack([self.decode_action(row) for row in actions], axis=0)
        return ActionChunk(actions=decoded)

    def apply_joint_limits(self, action: np.ndarray) -> np.ndarray:
        """按配置的关节限位裁剪动作。

        长度对齐策略：取 ``action`` / ``lower`` / ``upper`` 三者最短长度，
        **只裁剪公共前缀**，多出的维度原样保留。这样即使限位表比动作短
        （或不完整），也不会报错或误改其它维度。

        Args:
            action: 待裁剪的动作向量。

        Returns:
            裁剪后的副本；未配置限位时返回原数组（不拷贝）。
        """
        lower = self.config.joint_lower
        upper = self.config.joint_upper
        if lower is None or upper is None:
            return action  # 未配置限位 -> 不裁剪
        lower = np.asarray(lower, dtype=np.float32)
        upper = np.asarray(upper, dtype=np.float32)
        width = min(action.shape[0], lower.shape[0], upper.shape[0])
        clipped = action.copy()
        clipped[:width] = np.clip(action[:width], lower[:width], upper[:width])
        return clipped


# ---------------------------------------------------------------------------
# 组合适配器（控制回路实际使用的对象）
# ---------------------------------------------------------------------------


class EmbodimentAdapter:
    """组合观测适配器与动作适配器，作为控制回路的统一入口。

    持有一个默认 ``instruction``：控制回路通常只需在构造时给一次任务描述，
    之后每次编码观测都会自动带上，避免每步重复传参。

    用法:
        >>> adapter = EmbodimentAdapter(AdapterConfig(), instruction="pick up the cube")
        >>> obs = adapter.encode_observation(robot_obs)      # 上行
        >>> chunk = adapter.decode_chunk(model_actions)      # 下行
    """

    def __init__(
        self,
        config: Optional[AdapterConfig] = None,
        norm_stats: Optional[Dict[str, Any]] = None,
        instruction: str = "",
    ) -> None:
        """初始化并创建两个子适配器。

        Args:
            config: 共享配置（子适配器持有同一个对象引用）。
            norm_stats: 共享归一化统计量。
            instruction: 默认任务描述。
        """
        self.config = config or AdapterConfig()
        self.norm_stats = norm_stats or {}
        self.instruction = instruction
        self.observation = ObservationAdapter(self.config, self.norm_stats)
        self.action = ActionAdapter(self.config, self.norm_stats)

    def set_instruction(self, instruction: str) -> None:
        """更新编码观测时使用的任务描述（例如切换 episode 的任务）。"""
        self.instruction = instruction

    def set_norm_stats(self, norm_stats: Dict[str, Any]) -> None:
        """挂载（或替换）checkpoint 归一化统计量。

        会**同步写入两个子适配器**，保证上行/下行使用同一套统计量 ——
        这是正确性关键：如果只更新一个方向，动作会出现系统性偏移。
        """
        self.norm_stats = norm_stats or {}
        self.observation.norm_stats = self.norm_stats
        self.action.norm_stats = self.norm_stats

    def encode_observation(self, robot_observation: Dict[str, Any]) -> Observation:
        """把机器人观测编码为规范观测。

        若传入的字典没有 ``instruction``，则填入构造时设定的默认值
        （``setdefault``，不会覆盖调用方显式给出的指令）。

        Args:
            robot_observation: ``{"images": {...}, "state": ndarray, ...}``。

        Returns:
            规范观测对象。
        """
        robot_observation = dict(robot_observation)  # 浅拷贝，避免污染调用方字典
        robot_observation.setdefault("instruction", self.instruction)
        return self.observation.encode(robot_observation)

    def decode_action(self, action: np.ndarray) -> np.ndarray:
        """解码单个动作为机器人原生命令。"""
        return self.action.decode_action(action)

    def decode_chunk(self, action_chunk: np.ndarray) -> ActionChunk:
        """解码整个动作块为机器人原生命令序列。"""
        return self.action.decode_chunk(action_chunk)

    # -- 报告风格别名 -------------------------------------------------------
    # 技术报告里写作 UnitreeAdapter.encode_observation / decode_action，
    # 下面两个方法提供同义入口，便于对照报告阅读代码。
    def encode(self, images, state, instruction=None, task_name=None) -> Observation:
        """报告风格助手：等价于 ``encode_observation``，但参数摊平。

        Args:
            images: 相机名 → 画面。
            state: 原始 proprio。
            instruction: 任务描述；``None`` 时使用默认指令。
            task_name: 可选数据集键。
        """
        return self.observation.encode_observation(
            images=images,
            state=state,
            instruction=instruction if instruction is not None else self.instruction,
            task_name=task_name,
        )

    def decode(self, action: np.ndarray) -> np.ndarray:
        """报告风格助手：等价于 ``decode_action``。"""
        return self.decode_action(action)
