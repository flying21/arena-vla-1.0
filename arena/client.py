# 麻雀虽小智能科技（武汉）有限公司
"""策略客户端，以及仿真 / 真机共用的控制回路。

技术报告对应第 3.1 / 3.2 / 3.3 节。

核心结论：**控制回路对仿真与真机是完全一样的**

    obs = reset()
    while not task_finished:
        packet   = adapter.encode_observation(obs)   # 上行：机器人 → VLA 空间
        actions  = client.infer(packet)              # 一次服务器往返
        decoded  = adapter.decode_chunk(actions)     # 下行：VLA 空间 → 机器人
        for action in decoded:
            obs = robot.execute(action)              # 逐步执行

唯一的差异在 ``robot`` 对象：

* :class:`ArenaSimRobot` 包装 Isaac Lab / Arena 的 gymnasium 环境；
* :class:`UnitreeRobot` 对接宇树 SDK。

两者都实现 :class:`RobotInterface`，因此 :class:`ControlLoop` 完全不需要
知道自己在驱动仿真还是真机 —— 这正是"一脑两身"在代码层面的落点。

网络错误 vs 业务错误的区分（重要）
------------------------------------------------------------------
联调中实际出现过这种现象::

    POST /act  ->  200 OK
    body       ->  "error"

**HTTP 成功 ≠ 策略成功**。当时的原因是 Isaac Sim 相机链路缺陷导致全黑占位图，
视觉语言模型拿不到有效语义而返回 ``error``。这类问题属于**策略业务失败**，
不是网络失败；把它当成网络错误重试是无效的（重试还会得到同样的 error）。
控制回路因此采用"重试 + 安全零动作 + 连续错误阈值终止"三级降级策略，
详见 ``arena_g1_locomanip_pnp.py`` 里的 HTTP 闭环实现。
"""

from __future__ import annotations

import abc
import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from arena.adapter import EmbodimentAdapter
from arena.config import AdapterConfig, ClientConfig
from arena.types import ActionChunk, Observation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 网络客户端
# ---------------------------------------------------------------------------


class PolicyClientError(RuntimeError):
    """策略服务器不可达或响应格式非法时抛出。"""


class PolicyClient:
    """与 ARENA / OpenPI 的 ``/act`` 端点通信的 HTTP 客户端。

    具备"重试 + 线性退避"能力，把瞬时网络抖动挡在控制回路之外。
    """

    def __init__(self, config: Optional[ClientConfig] = None) -> None:
        """初始化。

        Args:
            config: 客户端配置；``None`` 时使用默认 :class:`ClientConfig`
                （指向 ``http://127.0.0.1:8777/act``，3 次重试，10s 超时）。
        """
        self.config = config or ClientConfig()

    def infer(self, observation: Observation, instruction: Optional[str] = None) -> ActionChunk:
        """发送一帧观测，取回动作块。

        重试策略：最多 ``config.retries`` 次，每次失败后按 ``0.05 × 尝试次数``
        秒退避（50ms、100ms、150ms…）。全部失败才抛
        :class:`PolicyClientError`。

        Args:
            observation: 规范观测。
            instruction: 覆盖观测内的指令；``None`` 时沿用观测自带的。

        Returns:
            :class:`~arena.types.ActionChunk`，``metadata`` 里带服务端上报的
            ``latency_s``（便于区分"网络慢"与"模型慢"）。

        Raises:
            PolicyClientError: 重试全部失败。
        """
        payload = self._build_payload(observation, instruction)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.retries + 1):
            try:
                data = self._post(payload)
                actions = self._extract_actions(data)
                return ActionChunk(actions=actions, metadata={"latency_s": data.get("latency_s")})
            except Exception as exc:  # noqa: BLE001 - 由下面的重试逻辑处理
                last_error = exc
                logger.warning("Policy request failed (attempt %d/%d): %s", attempt, self.config.retries, exc)
                time.sleep(0.05 * attempt)  # 线性退避，避免瞬间打爆服务端

        raise PolicyClientError(f"Failed to reach policy server: {last_error}")

    # -- 内部辅助 -----------------------------------------------------------
    def _build_payload(self, observation: Observation, instruction: Optional[str]) -> Dict[str, Any]:
        """把观测打包成 ``/act`` 期望的请求体。

        指令会**同时**写在观测内部与顶层：不同服务端实现读取位置不一致
        （ARENA 服务端优先读顶层，官方实现读 ``observations[0]``），
        两处都填可以兼容双方。
        """
        obs = observation.to_dict()
        if instruction is not None:
            obs["instruction"] = instruction
        return {"observations": [obs], "instruction": obs.get("instruction", "")}

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """执行 HTTP POST 并保证返回 dict。

        ``requests`` 会使用已被 ``json_numpy.patch()`` 替换过的
        ``json.dumps``，因此 payload 里的 numpy 数组能自动编码。
        """
        import requests  # 惰性导入

        response = requests.post(self.config.server_url, json=payload, timeout=self.config.timeout_s)
        response.raise_for_status()  # 非 2xx 直接抛，交给重试逻辑
        data = response.json()
        if not isinstance(data, dict):
            data = {"actions": data}  # 兼容只返回裸数组的服务端
        return data

    @staticmethod
    def _extract_actions(data: Dict[str, Any]) -> np.ndarray:
        """从响应里取出动作数组。

        依次尝试：``{"actions": ...}`` → ``{"__numpy__": ...}``（json_numpy 编码）。

        Raises:
            PolicyClientError: 两种格式都不匹配。
        """
        if "actions" in data:
            return np.asarray(data["actions"], dtype=np.float32)
        if "__numpy__" in data:
            try:
                import json_numpy

                return np.asarray(json_numpy.loads(json.dumps(data)), dtype=np.float32)
            except Exception:  # pragma: no cover
                pass
        raise PolicyClientError(f"Malformed response: {list(data)[:5]}")


# ---------------------------------------------------------------------------
# 机器人抽象接口
# ---------------------------------------------------------------------------


class RobotInterface(abc.ABC):
    """所有具身（仿真或真机）都必须实现的最小接口。

    只有三个抽象方法（``reset`` / ``get_observation`` / ``execute``），
    外加两个带默认实现的查询方法。接口刻意保持小，这样接入一台新机器人
    只需要写一个薄包装器，而不用改动框架任何其它部分。
    """

    @abc.abstractmethod
    def reset(self) -> Dict[str, Any]:
        """复位机器人并返回第一帧观测载荷。"""

    @abc.abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """返回 ``{"images": {相机名: 画面}, "state": ndarray}``。"""

    @abc.abstractmethod
    def execute(self, action: np.ndarray) -> None:
        """在机器人原生空间执行**单个**动作。"""

    def is_running(self) -> bool:
        """episode 是否应继续；默认始终继续（由外部步数上限控制）。"""
        return True

    def task_finished(self) -> bool:
        """任务是否已成功完成；默认永远未完成。"""
        return False


# ---------------------------------------------------------------------------
# 具身实现 1：Isaac Lab / Arena 仿真
# ---------------------------------------------------------------------------


class ArenaSimRobot(RobotInterface):
    """包装 Isaac Lab / Arena 的 gymnasium 环境。

    适用于任何遵循标准 gymnasium API 的环境，例如 ``unifolm-vla/arena_test.py``
    里用 ``lerobot`` 加载的 Arena 环境。

    健壮性设计：不同 gym / gymnasium 版本的 ``reset``/``step`` 返回值格式不同
    （2/4/5 元组），不同环境的观测键名也不同。因此这里用
    :meth:`_unwrap` 统一元组格式、用 :meth:`_search` 按候选键名顺序查找，
    让一个包装器能吃下尽可能多的环境。
    """

    #: 第三人称视角的候选观测键（按顺序查找，命中即用）。
    FULL_IMAGE_KEYS = ("full_image", "image", "agentview_image", "rgb", "head_image")
    #: 腕部相机的候选观测键。
    WRIST_IMAGE_KEYS = ("wrist_image", "robot0_eye_in_hand_image", "wrist", "hand_image")
    #: 本体感知状态的候选观测键。
    STATE_KEYS = ("state", "robot_state", "proprio", "joint_position", "observation.state")

    def __init__(self, env, instruction: str = "", max_steps: int = 1000) -> None:
        """初始化。

        Args:
            env: gymnasium 兼容环境。
            instruction: 默认任务描述。
            max_steps: episode 步数上限（即使任务未完成也会停）。
        """
        self.env = env
        self.instruction = instruction
        self.max_steps = max_steps
        self.steps = 0
        self._last_obs: Optional[Dict[str, Any]] = None

    def reset(self) -> Dict[str, Any]:
        """复位环境，缓存原始观测，返回规范化的观测载荷。"""
        result = self.env.reset()
        self._last_obs = self._unwrap(result)
        self.steps = 0
        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """按候选键名提取图像与状态，组装成 ARENA 载荷。

        找不到状态时返回**空数组**而不是报错：这样即使环境只提供图像，
        链路依然能跑（适配器会原样透传空状态）。

        Raises:
            RuntimeError: 未先调用 :meth:`reset`。
        """
        if self._last_obs is None:
            raise RuntimeError("reset() must be called before get_observation()")
        images = {}
        # 第三人称视角固定映射为 "head"（与适配器的命名约定一致）
        full = self._search(self._last_obs, self.FULL_IMAGE_KEYS)
        if full is not None:
            images["head"] = full
        wrist = self._search(self._last_obs, self.WRIST_IMAGE_KEYS)
        if wrist is not None:
            images["wrist"] = wrist
        state = self._search(self._last_obs, self.STATE_KEYS)
        if state is None:
            state = np.zeros(0, dtype=np.float32)
        return {"images": images, "state": np.asarray(state, dtype=np.float32).reshape(-1)}

    def execute(self, action: np.ndarray) -> None:
        """执行单个动作并更新缓存观测与步数计数。"""
        result = self.env.step(np.asarray(action, dtype=np.float32).reshape(-1))
        self._last_obs = self._unwrap(result)
        self.steps += 1

    def is_running(self) -> bool:
        """未达步数上限且任务未完成时继续。"""
        return self.steps < self.max_steps and not self.task_finished()

    def task_finished(self) -> bool:
        """检查终止标记。

        依次检查 ``terminated`` / ``truncated`` / ``done`` / ``success``。
        注意 ``np.asarray(...).any()``：环境可能返回**批量**布尔数组
        （Isaac Lab 常见），只要有一个为真就认为结束。
        """
        if self._last_obs is None:
            return False
        for key in ("terminated", "truncated", "done", "success"):
            value = self._last_obs.get(key)
            if value is not None:
                arr = np.asarray(value)
                if bool(arr.any()):
                    return True
        return False

    # -- 内部辅助 -----------------------------------------------------------
    @staticmethod
    def _unwrap(result: Any) -> Dict[str, Any]:
        """把 gym / gymnasium 的 ``reset``/``step`` 返回值统一成扁平字典。

        支持三种元组格式：

        * 5 元组 ``(obs, reward, terminated, truncated, info)`` —— gymnasium 0.29+；
        * 4 元组 ``(obs, reward, done, info)`` —— 经典 gym / 旧版 gymnasium；
        * 2 元组 ``(obs, info)`` —— reset 的常见返回。

        使用 ``setdefault`` 而不是直接赋值，是为了**不覆盖**观测字典里已有的
        同名字段（以观测为准）。
        """
        if isinstance(result, tuple):
            if len(result) == 5:  # (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = result
                merged = dict(obs) if isinstance(obs, dict) else {"state": obs}
                merged.setdefault("terminated", terminated)
                merged.setdefault("truncated", truncated)
                merged.setdefault("reward", reward)
                return merged
            if len(result) == 4:  # (obs, reward, done, info)
                obs, reward, done, info = result
                merged = dict(obs) if isinstance(obs, dict) else {"state": obs}
                merged.setdefault("done", done)
                merged.setdefault("reward", reward)
                return merged
            # (obs, info)
            obs, info = result[0], result[1] if len(result) > 1 else {}
            merged = dict(obs) if isinstance(obs, dict) else {"state": obs}
            if isinstance(info, dict):
                merged.update(info)  # info 里的 success 等字段一并并入
            return merged
        if isinstance(result, dict):
            return result
        # 兜底：裸数组当作状态
        return {"state": np.asarray(result, dtype=np.float32).reshape(-1)}

    @staticmethod
    def _search(observation: Dict[str, Any], keys) -> Optional[np.ndarray]:
        """按候选键名顺序查找第一个存在且非 None 的值，转成 numpy 返回。"""
        for key in keys:
            if key in observation and observation[key] is not None:
                return np.asarray(observation[key])
        return None


# ---------------------------------------------------------------------------
# 具身实现 2：Unitree 真机
# ---------------------------------------------------------------------------


class UnitreeRobot(RobotInterface):
    """包装宇树 SDK / 部署控制器的适配器。

    具体控制器以 ``sdk`` **注入**，它需要提供 ``reset()``、``get_state()``、
    ``get_camera(name)`` 与 ``execute(action)``。

    这样设计的好处：ARENA 不绑定任何特定版本的宇树 SDK。SDK 升级、
    或换成官方 ``unitree_deploy`` 客户端，只需换注入对象，框架代码不动。
    """

    def __init__(self, sdk, instruction: str = "", camera_names=("head", "wrist")) -> None:
        """初始化。

        Args:
            sdk: 机器人 SDK / 部署控制器实例。
            instruction: 默认任务描述。
            camera_names: 需要拉取的相机名列表。
        """
        self.sdk = sdk
        self.instruction = instruction
        self.camera_names = camera_names
        self._running = True

    def reset(self) -> Dict[str, Any]:
        """复位 SDK（若其支持）并返回当前观测。"""
        if hasattr(self.sdk, "reset"):
            self.sdk.reset()
        self._running = True
        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """拉取所有相机的当前帧与本体状态。

        相机返回 ``None`` 时跳过该相机（允许某些硬件缺装相机）。
        """
        images = {}
        for name in self.camera_names:
            frame = self.sdk.get_camera(name)
            if frame is not None:
                images[name] = frame
        state = self.sdk.get_state()
        return {"images": images, "state": np.asarray(state, dtype=np.float32).reshape(-1)}

    def execute(self, action: np.ndarray) -> None:
        """把单个动作下发给 SDK 执行。"""
        self.sdk.execute(np.asarray(action, dtype=np.float32).reshape(-1))

    def is_running(self) -> bool:
        """优先询问 SDK；SDK 未提供该能力时使用本地运行标志。"""
        if hasattr(self.sdk, "is_running"):
            return bool(self.sdk.is_running())
        return self._running

    def stop(self) -> None:
        """停止运行：置本地标志并（若支持）通知 SDK 停止。

        真机场景下这是**安全出口**，用于异常/急停路径。
        """
        self._running = False
        if hasattr(self.sdk, "stop"):
            self.sdk.stop()


# ---------------------------------------------------------------------------
# 统一闭环
# ---------------------------------------------------------------------------


class ControlLoop:
    """与具身无关的闭环控制器（技术报告第 3.3 节）。

    只依赖三个抽象：:class:`RobotInterface`、``PolicyClient``（任何带 ``infer``
    的对象）与 :class:`~arena.adapter.EmbodimentAdapter`。
    """

    def __init__(
        self,
        robot: RobotInterface,
        client: PolicyClient,
        adapter: EmbodimentAdapter,
        instruction: str = "",
        max_episodes: int = 1,
    ) -> None:
        """初始化。

        Args:
            robot: 具身实现（仿真或真机）。
            client: 策略客户端；只要求有 ``infer(observation, instruction)``
                方法，因此也可以传入进程内直连后端的轻量客户端。
            adapter: 具身适配器（上行编码 + 下行解码）。
            instruction: 默认任务描述。
            max_episodes: :meth:`run` 要跑的 episode 数。
        """
        self.robot = robot
        self.client = client
        self.adapter = adapter
        self.instruction = instruction
        self.max_episodes = max_episodes

    def run_episode(self, instruction: Optional[str] = None) -> Dict[str, Any]:
        """运行单个 episode 并返回统计摘要。

        循环结构（对应本模块开头伪代码）：

        1. 复位机器人；
        2. 取观测 → 适配器编码 → 客户端推理 → 适配器解码；
        3. **逐个执行**解码后的动作，每步都检查任务是否已完成
           （以便在成功瞬间立即停止，避免多余的"画蛇添足"动作）。

        Args:
            instruction: 覆盖本次 episode 的任务描述；``None`` 时用默认值。

        Returns:
            ``{"success": bool}``。注意 ``success`` 取自
            :meth:`RobotInterface.task_finished`，因此它反映的是**机器人环境
            认可的终止/成功标记**，而不是外部评估。
        """
        if instruction is not None:
            self.adapter.set_instruction(instruction)
        task = instruction or self.instruction

        self.robot.reset()
        while self.robot.is_running() and not self.robot.task_finished():
            robot_obs = self.robot.get_observation()
            robot_obs.setdefault("instruction", task)  # 不覆盖环境自带的指令
            observation = self.adapter.encode_observation(robot_obs)
            chunk = self.client.infer(observation, instruction=task)
            decoded = self.adapter.decode_chunk(chunk.actions)
            for action in decoded:
                self.robot.execute(action)
                if self.robot.task_finished():
                    break  # 成功即停，不再执行块内剩余动作
        return {"success": bool(self.robot.task_finished())}

    def run(self, instruction: Optional[str] = None) -> List[Dict[str, Any]]:
        """连续运行 ``max_episodes`` 个 episode，收集每个的摘要。"""
        return [self.run_episode(instruction) for _ in range(self.max_episodes)]
