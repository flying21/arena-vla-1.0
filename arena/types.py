# 麻雀虽小智能科技（武汉）有限公司
"""ARENA 全局统一数据契约：规范观测 ``Observation`` 与规范动作块 ``ActionChunk``。

本模块是整个框架**最底层、最稳定**的一层，不依赖任何其它 arena 子模块
（只依赖 numpy），因此可以被 Server / Client / Adapter / Robot 任意一方安全导入。

技术报告定义的规范接口::

    O_t       = {I_head, I_wrist, S_t, L}           # 单帧观测
    A_{t:t+H} = [a_t, a_{t+1}, ..., a_{t+H}]        # 长度为 H 的动作块

设计意图（为什么需要这一层）
------------------------------------------------------------------
不同机器人有着不同数量的关节、相机、坐标系与控制频率；不同 VLA 模型也有
各自的输入输出格式。如果让上层模块互相猜测对方的数据格式，那么每换一个
模型、每换一个机器人，都要重写整条链路。

因此这里定义一个**唯一的"中间语言"**::

    机器人原生数据 --EmbodimentAdapter--> Observation --> 策略服务
                       (见 arena/adapter.py)

    策略输出(动作块) --EmbodimentAdapter--> 机器人原生动作
                       (见 arena/adapter.py)

有了这一层：Server 不需要知道对面是 G1 还是 GR1；Adapter 不需要知道
模型是 OpenPI 还是 UnifoLM-VLA；新增机器人只需实现 RobotInterface。

术语约定
------------------------------------------------------------------
* **第三人称视角 / full view**：头戴相机或场景相机拍到的"全局视角"，
  规范键名写作 ``"full"``。代码为兼容各种数据源接受若干别名，
  见 :data:`FULL_IMAGE_KEY_ALIASES`。
* **腕部相机 / wrist view**：装在末端执行器上的"眼在手上"视角，
  键名中只要含 ``"wrist"`` 就会被识别（见 :attr:`Observation.wrist_images`）。
* **proprio（本体感知）**：机器人自身的关节角度 / 末端位姿等状态向量，
  规范命名为 ``state``。

序列化契约（重要）
------------------------------------------------------------------
:meth:`Observation.to_dict` 会把第三人称视角**统一写成 ``"full_image"`` 键**，
因为官方 UnifoLM-VLA 服务端（``deployment/model_server/run_real_eval_server.py``）
硬编码读取该键名。由此带来一个必须注意的后果：

* 传输是**单向规范化**的：调用方原本用的别名（``"head"`` / ``"image"`` / …）
  不会被送到对端；
* :meth:`Observation.from_dict` 读回来时，第三人称视角一律叫 ``images["full"]``，
  因此 ``from_dict(to_dict(obs))`` **不能逐字保留别名**，但能保证
  "每台相机只出现一次、其余相机的键名原样保留"。

该行为有回归测试锁定，见 ``tests/test_arena.py`` 中的
``test_observation_round_trip_has_no_duplicate_camera``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 模块级常量
# ---------------------------------------------------------------------------

#: UnifoLM-VLA 系列模型期望的输入图像边长（正方形，单位像素）。
#: Adapter 会把所有相机画面缩放到 ``(224, 224, 3)``，见 arena/adapter.py::resize_image。
VLA_IMAGE_SIZE = 224

#: 被视为"第三人称视角"的相机键名别名集合。
#:
#: 注意 ``"full_image"`` 本身也在列表里：它是 :meth:`Observation.to_dict` 写出的
#: 线上键名，必须在 :meth:`Observation.from_dict` 读回来时被识别为"别名"，
#: 否则同一帧会被同时存成 ``images["full"]`` 与 ``images["full_image"]`` 两份。
FULL_IMAGE_KEY_ALIASES = ("full", "full_image", "head", "image", "agentview_image")

#: 不属于相机图像的 payload 字段，反序列化时需要跳过。
_NON_IMAGE_KEYS = ("state", "instruction", "task_name", "metadata")


def _is_full_image_key(key: str) -> bool:
    """判断相机键名是否属于"第三人称视角"的别名。

    Args:
        key: 相机键名，例如 ``"head"`` / ``"wrist"`` / ``"full_image"``。

    Returns:
        True 表示该键是第三人称视角的别名（多个键可能指向同一张图）。
    """
    return key in FULL_IMAGE_KEY_ALIASES


# ---------------------------------------------------------------------------
# 观测
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    """规范 VLA 观测 ``O_t = {I_head, I_wrist, S_t, L}``。

    这是机器人侧与策略侧之间传递"当前看到了什么"的唯一载体。

    Attributes:
        images: 相机名 → RGB 图像 ``(H, W, 3)`` ``uint8`` 的映射。
            约定 ``"head"``/``"full"`` 为第三人称视角，键名含 ``"wrist"`` 的
            为眼在手上视角。允许为空（纯状态观测）。
        state: 本体感知向量（关节位置 / 末端位姿等），一维 ``float32``。
            是否已归一化取决于 Adapter 是否配置了 ``norm_stats``。
        instruction: 自然语言任务描述，例如 ``"pick up the cube"``。
        task_name: 可选的数据集键，用于选择对应的归一化统计量
            （多任务 checkpoint 的 ``dataset_statistics.json`` 按任务分块）。
        metadata: 自由附加字段，原样透传，框架自身不解释其内容。

    示例:
        >>> import numpy as np
        >>> obs = Observation(
        ...     images={"head": np.zeros((224, 224, 3), np.uint8)},
        ...     state=np.zeros(8, np.float32),
        ...     instruction="pick up the cube",
        ... )
        >>> obs.full_image.shape
        (224, 224, 3)
    """

    images: Dict[str, np.ndarray]
    state: np.ndarray
    instruction: str = ""
    task_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- 便捷访问器 ---------------------------------------------------------
    @property
    def full_image(self) -> np.ndarray:
        """返回第三人称视角图像。

        查找顺序为 :data:`FULL_IMAGE_KEY_ALIASES`（``full`` → ``full_image`` →
        ``head`` → ``image`` → ``agentview_image``）；若一个都没有，则退化为
        "第一台可用相机"，以免调用方因为命名差异直接崩掉。

        Returns:
            ``(H, W, 3)`` 的图像数组。

        Raises:
            KeyError: 该观测里完全没有图像。
        """
        for key in FULL_IMAGE_KEY_ALIASES:
            if key in self.images:
                return self.images[key]
        # 兜底：取任意第一台相机，保证下游至少能跑通
        if not self.images:
            raise KeyError("Observation contains no images")
        return next(iter(self.images.values()))

    @property
    def wrist_images(self) -> List[np.ndarray]:
        """返回所有腕部（眼在手上）图像，保持插入顺序。

        判定规则很简单：键名中包含 ``"wrist"`` 即视为腕部相机。
        """
        return [img for key, img in self.images.items() if "wrist" in key]

    @property
    def all_images(self) -> List[np.ndarray]:
        """返回 ``[第三人称视角, *所有腕部视角]``。

        帧序即模型输入序：**全景点在前、腕部点在后**，与
        ``deployment/model_server/run_real_eval_server.py`` 中"先拼
        ``full_image`` 再扩展 ``wrist``"的顺序保持一致。

        别名键按数组对象身份去重：``{"full": img, "full_image": img}``
        只产出一帧，而不会把同一张图算两次。
        """
        images: List[np.ndarray] = []
        seen: set = set()
        for key, img in self.images.items():
            if _is_full_image_key(key):
                if id(img) in seen:
                    continue  # 同一张图被多个别名引用，只算一帧
                seen.add(id(img))
            images.append(img)
        return images

    def to_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 传输的字典（图像仍是 numpy 数组）。

        输出结构::

            {
              "full_image": ndarray,       # 第三人称视角，固定使用该键名
              "state":      ndarray,       # float32
              "instruction": str,
              "task_name":   str | 缺省,    # 仅当非 None 时写入
              "metadata":    dict | 缺省,   # 仅当非空时写入
              "<其它相机名>": ndarray,       # 例如 "wrist"
            }

        注意:
            第三人称视角会被**规范化**为 ``"full_image"``；调用方原本用的别名
            （``"head"`` 等）不会保留 —— 这是为了满足官方服务端的硬编码读法。
            其余相机保留自己的键名，:meth:`from_dict` 会原样还原。

        传输层:
            真正的 numpy → JSON 编码由 ``json_numpy`` 完成（把数组编码成带
            ``"__numpy__"`` 标记的字符串），见 ``arena/server.py``。
        """
        payload: Dict[str, Any] = {
            "full_image": np.asarray(self.full_image, dtype=np.uint8),
            "state": np.asarray(self.state, dtype=np.float32),
            "instruction": self.instruction,
        }
        for key, img in self.images.items():
            if _is_full_image_key(key):
                continue  # 上面已写成 "full_image"，此处跳过以免重复传图
            payload[key] = np.asarray(img, dtype=np.uint8)
        if self.task_name is not None:
            payload["task_name"] = self.task_name
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Observation":
        """从解码后的 payload 还原 :class:`Observation`。

        还原规则：

        1. 跳过 :data:`_NON_IMAGE_KEYS` 与所有别名键；
        2. 其余**三维**数组一律视为相机，按原键名存入 ``images``；
        3. 第三人称视角统一存为 ``images["full"]``（线上只有一个
           ``"full_image"`` 条目，别名本身不传输，故无法逐字还原）。

        与旧版的差别（已修复的缺陷）:
            旧实现把 payload 里的 ``"full_image"`` 也当成一台具名相机收进
            ``images``，**再**额外插入 ``images["full"]``，导致同一帧被存两份：
            ``{'head','wrist'}`` 往返后变成 ``{'full','full_image','wrist'}``。
            现在改为"先排除别名、再单独还原"，并保证不产生重复帧。

        Args:
            payload: :meth:`to_dict` 的产物，或对端用 ``json_numpy.loads``
                解出的等价字典。缺失 ``state``/``instruction`` 时分别退化为
                空数组与空串。

        Returns:
            还原后的 :class:`Observation`。
        """
        images: Dict[str, np.ndarray] = {}
        for key, value in payload.items():
            if _is_full_image_key(key) or key in _NON_IMAGE_KEYS:
                continue
            # 只把三维数组当图像，避免把误入的二维矩阵成像
            if isinstance(value, np.ndarray) and value.ndim == 3:
                images[key] = value

        # 第三人称视角：优先取规范键名，其次兼容"只带别名"的旧版 payload
        full = payload.get("full_image")
        if full is None:
            full = next((payload[k] for k in FULL_IMAGE_KEY_ALIASES if k in payload), None)
        if full is not None:
            images["full"] = np.asarray(full)

        return cls(
            images=images,
            state=np.asarray(payload.get("state", []), dtype=np.float32),
            instruction=str(payload.get("instruction", "")),
            task_name=payload.get("task_name"),
            metadata=dict(payload.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# 动作块
# ---------------------------------------------------------------------------


@dataclass
class ActionChunk:
    """一段"策略推理"与"逐步执行"之间的解耦载体。

    规范定义 ``A_{t:t+H} = [a_t, ..., a_{t+H}]``，形状 ``(H, action_dim)``。

    为什么叫"块"（chunk）
        一次 VLA 推理能够预测未来 H 步动作。一次性取回 H 步、再逐步下发给
        机器人，可以显著降低"每走一步都要等一次大模型"的网络抖动 —— 这正是
        本项目引入动作块的核心动机。

    Attributes:
        actions: ``(H, D)`` ``float32``。传入一维向量时会自动升为 ``(1, D)``。
        metadata: 附加信息，例如 ``{"latency_s": 0.05}``（服务端回传的推理延迟）。

    示例:
        >>> chunk = ActionChunk(actions=np.zeros(7))
        >>> len(chunk), chunk.action_dim
        (1, 7)
    """

    actions: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """统一 dtype 并把一维输入升成二维，保证后续索引语义稳定。"""
        self.actions = np.asarray(self.actions, dtype=np.float32)
        if self.actions.ndim == 1:
            # 单步动作 (D,) -> (1, D)，让 actions[i] 永远表示"第 i 步"
            self.actions = self.actions[None, :]

    def __len__(self) -> int:
        """返回块长度 H（步数）。"""
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        """返回单步动作维度 D。"""
        return int(self.actions.shape[1])

    def __iter__(self):
        """支持 ``for action in chunk`` 逐步遍历，每项形状为 ``(D,)``。"""
        return iter(self.actions)

    def __getitem__(self, index):
        """支持 ``chunk[i]`` 取第 i 步动作。"""
        return self.actions[index]
