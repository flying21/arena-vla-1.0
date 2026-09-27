# 麻雀虽小智能科技（武汉）有限公司
"""Sim-to-Real 域随机化课程。

技术报告对应第 5 节。

随机化后的环境分布记为::

    E ~ P(L, T, C, M, mu, N)

其中

* ``L`` —— 光照 / 渲染随机化
* ``T`` —— 纹理 / 外观随机化
* ``C`` —— 相机位姿与内参随机化
* ``M`` —— 质量 / 摩擦 / 执行器动力学随机化
* ``mu`` —— 观测噪声
* ``N`` —— 延迟 / 通信抖动

四级课程逐步把策略暴露到这些因素上，使仿真中训练的策略能迁移到真机。

设计理念："不是把仿真做得完美，而是让它不完美得足够多"
------------------------------------------------------------------
固定光照、固定相机、固定摩擦力的仿真很容易让策略"记住唯一一个世界"。
本模块的思路是**主动制造大量可控扰动**，逼模型学到在多种环境中都成立的
动作规律，从而逐步逼近真机分布。

实现边界（诚实说明）
------------------------------------------------------------------
============  ==========  ====================================================
因子           是否实现     说明
============  ==========  ====================================================
``L`` 光照      ✅ 直接施加  :meth:`Randomizer.randomize_image` 亮度乘子
``T`` 纹理      ✅ 直接施加  :meth:`Randomizer.randomize_image` 对比度乘子
``C`` 相机      ✅ 直接施加  :meth:`Randomizer.randomize_camera`（图像空间）
``mu`` 噪声     ✅ 直接施加  :meth:`Randomizer.randomize_state`
``N`` 延迟      ✅ 直接施加  :meth:`Randomizer.delay_action`
``M`` 动力学    ⚠️ 只出参数  :meth:`Randomizer.randomize_dynamics` 返回缩放系数，
                             **施加必须由仿真器完成**
============  ==========  ====================================================

为什么 ``M`` 只产生参数
    本模块刻意**零依赖**（只用 numpy），因此它能改写"图像字节"，却无法写入
    物理引擎的质量/摩擦属性。声明"输出参数、由引擎施加"是这种架构下唯一
    诚实的做法。具体施加代码见 :class:`DynamicsRandomization` 的文档字符串。

为什么 ``C`` 用图像空间近似
    裁剪 + 平移 + 传感器噪声可以模拟焦距与相机姿态变化对画面的影响，
    但**无法复现真正的透视/视差变化** —— 那必须由渲染器完成。这样做换来的是
    "可包裹任意具身（Isaac Lab / MuJoCo / 回放缓冲）而不需要 OpenGL 上下文"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from arena.config import Sim2RealConfig


# ---------------------------------------------------------------------------
# 随机化档案
# ---------------------------------------------------------------------------


@dataclass
class DomainRandomization:
    """一份具体的域随机化档案（某个等级对应一组幅度）。

    所有幅度字段都是**最大相对偏差** ``m``，而不是绝对物理量。
    具体采样公式见 :data:`CURRICULUM` 的注释与 :class:`Randomizer` 各方法。

    Attributes:
        level: 等级编号 1..4。
        lighting: 亮度乘子的最大偏差。
        texture: 对比度乘子的最大偏差。
        camera: 相机扰动幅度（同时约束裁剪、平移与噪声）。
        dynamics: 动力学缩放系数的最大偏差。
        observation_noise: proprio 高斯噪声标准差（归一化空间）。
        action_delay_steps: 动作延迟的控制步数。
    """

    level: int = 1
    lighting: float = 0.0
    texture: float = 0.0
    camera: float = 0.0
    dynamics: float = 0.0
    observation_noise: float = 0.0
    action_delay_steps: int = 0

    def describe(self) -> Dict[str, Any]:
        """转成可打印 / 可记录的普通字典。"""
        return {
            "level": self.level,
            "lighting": self.lighting,
            "texture": self.texture,
            "camera": self.camera,
            "dynamics": self.dynamics,
            "observation_noise": self.observation_noise,
            "action_delay_steps": self.action_delay_steps,
        }


#: 1–4 级课程定义（唯一权威口径，文档中的数值表以此为准）。
#:
#: 每个量都是**最大相对偏差** ``m``，而非绝对物理量：
#:
#: * ``lighting`` / ``texture`` —— 亮度 / 对比度乘子采样自 ``1 + U(-m, m)``；
#: * ``camera`` —— 约束裁剪比例与平移量，并把传感器噪声按 ``m * 8`` 灰度级缩放；
#: * ``dynamics`` —— 质量 / 摩擦 / 执行器增益 / 阻尼乘子采样自 ``1 + U(-m, m)``
#:   （并按 :data:`MAX_DYNAMICS_MAGNITUDE` 裁剪）；
#: * ``observation_noise`` —— proprio 高斯噪声标准差，落在适配器归一化后的
#:   ``[-1, 1]`` 空间（**不是物理单位**）；
#: * ``action_delay_steps`` —— 整数控制步延迟。
#:
#: ``DR-L1`` 刻意保持**完全中性**：它是"链路到底通不通"的基线配置，
#: 若此处任何幅度大于 0，教科书式的课程阶梯就无法复现。
CURRICULUM: Dict[int, DomainRandomization] = {
    1: DomainRandomization(
        level=1,
        lighting=0.0,
        texture=0.0,
        camera=0.0,
        dynamics=0.0,
        observation_noise=0.0,
        action_delay_steps=0,
    ),
    2: DomainRandomization(
        level=2,
        lighting=0.15,
        texture=0.15,
        camera=0.05,
        dynamics=0.1,
        observation_noise=0.01,
        action_delay_steps=1,
    ),
    3: DomainRandomization(
        level=3,
        lighting=0.3,
        texture=0.3,
        camera=0.15,
        dynamics=0.25,
        observation_noise=0.03,
        action_delay_steps=2,
    ),
    4: DomainRandomization(
        level=4,
        lighting=0.5,
        texture=0.5,
        camera=0.3,
        dynamics=0.5,
        observation_noise=0.05,
        action_delay_steps=3,
    ),
}


# ---------------------------------------------------------------------------
# 随机化采样结果
# ---------------------------------------------------------------------------


@dataclass
class CameraRandomization:
    """一次相机域扰动的采样结果（2–4 级使用）。

    ``zoom`` 模拟焦距/距离变化，``shift`` 模拟相机支架转动，``noise_sigma``
    是传感器噪声。扰动在**图像空间**施加，使随机化器保持零依赖：任意具身都能
    包住自己的相机链路，而不需要 OpenGL 上下文。

    局限: 无法复现真正的透视变化（视差），那必须由仿真器提供。

    Attributes:
        zoom: 缩放因子，``>= 1`` 表示"放大"（裁剪后再重采样，不产生黑边）。
        shift_x: x 方向归一化平移，范围约 ``[-1, 1]``。
        shift_y: y 方向归一化平移。
        noise_sigma: 加性高斯噪声标准差（灰度级）。
    """

    zoom: float = 1.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    noise_sigma: float = 0.0

    @property
    def is_identity(self) -> bool:
        """是否为恒等变换（不做任何改动）。

        ``zoom`` 用 1e-9 的容差比较，因为它是浮点乘子；其余字段与 0 精确比较。
        """
        return (
            abs(self.zoom - 1.0) < 1e-9
            and self.shift_x == 0.0
            and self.shift_y == 0.0
            and self.noise_sigma == 0.0
        )

    def describe(self) -> Dict[str, float]:
        """转成可打印 / 可记录的普通字典。"""
        return {
            "zoom": self.zoom,
            "shift_x": self.shift_x,
            "shift_y": self.shift_y,
            "noise_sigma": self.noise_sigma,
        }


@dataclass
class DynamicsRandomization:
    """一次动力学域扰动的采样结果（2–4 级使用）。

    这些是**物理参数**，因此本类只负责"产生"它们，施加需要引擎参与。
    以 Isaac Lab 为例::

        # prim path -> physics view（质量 / 摩擦 / 阻尼）
        view = scene[asset_name].root_physx_view
        view.set_masses(view.get_masses() * d.mass_scale, indices)
        view.set_material_properties(
            view.get_material_properties() * [1.0, d.friction_scale, 1.0], indices
        )

    把采样放在这里（而不是放在环境里）的好处是：**所有任务共享同一份课程定义**，
    不同任务的扰动分布不会各自漂移。见 :meth:`Randomizer.randomize_dynamics`。

    Attributes:
        mass_scale: 质量缩放系数。
        friction_scale: 摩擦系数缩放。
        actuator_gain_scale: 执行器增益缩放（模拟电机力矩偏差）。
        damping_scale: 阻尼缩放。
    """

    mass_scale: float = 1.0
    friction_scale: float = 1.0
    actuator_gain_scale: float = 1.0
    damping_scale: float = 1.0

    @property
    def is_identity(self) -> bool:
        """是否四项全部为 1.0（即不做动力学扰动）。"""
        return (
            self.mass_scale == 1.0
            and self.friction_scale == 1.0
            and self.actuator_gain_scale == 1.0
            and self.damping_scale == 1.0
        )

    def describe(self) -> Dict[str, float]:
        """转成可打印 / 可记录的普通字典。"""
        return {
            "mass_scale": self.mass_scale,
            "friction_scale": self.friction_scale,
            "actuator_gain_scale": self.actuator_gain_scale,
            "damping_scale": self.damping_scale,
        }


#: 动力学抖动幅度上限取 ±50 %：保证扰动**永远不会**产生非物理的（负数）
#: 质量或摩擦系数。用户即使把 profile 的 dynamics 写成 99，也会被裁到这里。
MAX_DYNAMICS_MAGNITUDE = 0.5
#: 缩放系数下界（0.5 = 减半）。
MIN_DYNAMICS_SCALE = 0.5
#: 缩放系数上界（1.5 = 增加 50 %）。
MAX_DYNAMICS_SCALE = 1.5


# ---------------------------------------------------------------------------
# 随机化器
# ---------------------------------------------------------------------------


class Randomizer:
    """把一份 :class:`DomainRandomization` 档案施加到观测 / 动作上。

    本类刻意**零依赖**：只扰动 numpy 数组，因此可以包裹任意具身
    （Isaac Lab、MuJoCo，或一个回放缓冲区），而不需要接触物理引擎本身。
    这带来的直接好处是：单元测试无需 GPU / 仿真即可覆盖全部随机化逻辑。

    ``E ~ P(L, T, C, M, mu, N)`` 各因子的覆盖情况见模块文档字符串。
    """

    def __init__(self, profile: DomainRandomization, seed: int = 42) -> None:
        """初始化。

        Args:
            profile: 要施加的随机化档案（通常取自 :data:`CURRICULUM`）。
            seed: 随机种子；相同种子 + 相同调用顺序 => 完全相同的扰动序列。
        """
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        #: 动作延迟用的环形缓冲（保存最近若干步动作）。
        self._delay_buffer: List[np.ndarray] = []

    # -- 观测侧随机化 -------------------------------------------------------
    def randomize_image(
        self,
        image: np.ndarray,
        photometry: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """扰动亮度/对比度，模拟光照与纹理变化（因子 ``L`` / ``T``）。

        算法：先减去图像均值再乘对比度、最后加回"均值 × 亮度"，
        这样**调整亮度不会同时改变对比度**（两个因子解耦）。

        Args:
            image: 输入画面。
            photometry: 可选的 ``(brightness, contrast)``，来自
                :meth:`sample_photometry`。传入它可以让多台相机共享同一组
                光照设置；不传则本方法自己采样一次。

        Returns:
            ``uint8`` 图像；档案中光照与纹理都为 0 时**原样返回输入**。
        """
        if self.profile.lighting <= 0 and self.profile.texture <= 0:
            return image
        if photometry is None:
            photometry = self.sample_photometry()
        brightness, contrast = photometry
        image = np.asarray(image, dtype=np.float32)
        mean = image.mean()
        randomized = (image - mean) * contrast + mean * brightness
        return np.clip(randomized, 0, 255).astype(np.uint8)

    def sample_camera(self) -> CameraRandomization:
        """为当前等级采样一次相机扰动（因子 ``C``）。

        ``profile.camera`` 是最大**相对**偏差：它同时约束裁剪比例、平移量，
        并按 ``m * 8`` 缩放传感器噪声。1 级（或任何 ``camera <= 0`` 的档案）
        返回恒等采样。

        设计选择: ``zoom`` 只放大不缩小（``1 + U(0, m)``）。这样重采样永远发生
            在原图**内部**，无需边界填充，输出尺寸与输入严格一致。

        Returns:
            采样得到的 :class:`CameraRandomization`。
        """
        magnitude = max(0.0, float(self.profile.camera))
        if magnitude <= 0.0:
            return CameraRandomization()
        # zoom 至少为 1.0，保证只需裁剪 + 平移 + 重采样，不引入黑边
        zoom = 1.0 + self.rng.uniform(0.0, magnitude)
        shift_x = self.rng.uniform(-magnitude, magnitude)
        shift_y = self.rng.uniform(-magnitude, magnitude)
        noise_sigma = magnitude * 8.0  # 例如 0.15 -> 约 1.2 个灰度级
        return CameraRandomization(
            zoom=float(zoom),
            shift_x=float(shift_x),
            shift_y=float(shift_y),
            noise_sigma=float(noise_sigma),
        )

    def randomize_camera(
        self,
        image: np.ndarray,
        camera: Optional[CameraRandomization] = None,
        noise: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """对单帧施加相机扰动（图像空间近似）。

        实现：把输出像素坐标反向映射到原图的归一化坐标
        ``coord * keep + shift``（``keep = 1/zoom``），裁剪到 ``[0, 1]``，
        再做双线性重采样。因为采样点始终被夹在原图范围内，
        **输出尺寸与输入完全一致，且不会出现黑边**。

        Args:
            image: 输入画面。
            camera: 相机扰动；``None`` 时现场采样一次。
            noise: 可选的**预先采好的**加性噪声场。传入它可以让多台相机
                共享同一次传感器采样，避免各相机噪声互不相关。

        Returns:
            ``uint8`` 图像；恒等扰动时**原样返回输入对象**。
        """
        original = np.asarray(image)
        if camera is None:
            camera = self.sample_camera()
        if camera.is_identity:
            return original
        image = original.astype(np.float32)

        height, width = image.shape[:2]
        # 每轴保留的画面比例（magnitude <= 0.5 时 >= 1/2）
        keep = 1.0 / max(camera.zoom, 1.0001)
        rows = np.clip(np.linspace(0.0, 1.0, height) * keep + camera.shift_y, 0.0, 1.0)
        cols = np.clip(np.linspace(0.0, 1.0, width) * keep + camera.shift_x, 0.0, 1.0)

        # 映射到像素坐标并取四个邻居 + 双线性权重
        src_r = rows * (height - 1)
        src_c = cols * (width - 1)
        r0 = np.floor(src_r).astype(np.int64)
        c0 = np.floor(src_c).astype(np.int64)
        r1 = np.clip(r0 + 1, 0, height - 1)
        c1 = np.clip(c0 + 1, 0, width - 1)
        wr = (src_r - r0)[:, None, None]
        wc = (src_c - c0)[None, :, None]

        top = image[r0][:, c0] * (1.0 - wc) + image[r0][:, c1] * wc
        bottom = image[r1][:, c0] * (1.0 - wc) + image[r1][:, c1] * wc
        sampled = top * (1.0 - wr) + bottom * wr

        # 噪声优先用外部传入的共享噪声场，否则现场采样（单相机独立噪声）
        if noise is not None:
            sampled = sampled + noise
        elif camera.noise_sigma > 0:
            sampled = sampled + self.rng.normal(0.0, camera.noise_sigma, size=sampled.shape)
        return np.clip(sampled, 0, 255).astype(np.uint8)

    def randomize_state(self, state: np.ndarray) -> np.ndarray:
        """给 proprio 叠加高斯噪声（因子 ``mu``）。

        量纲说明: ``profile.observation_noise`` 是**归一化 VLA 状态空间**
            （适配器把 proprio 映射到 ``[-1, 1]``）里的标准差，**不是物理单位**。

        Args:
            state: 本体感知向量。

        Returns:
            加噪后的 ``float32`` 状态；噪声为 0 时原样返回。
        """
        state = np.asarray(state, dtype=np.float32)
        if self.profile.observation_noise <= 0:
            return state
        noise = self.rng.normal(0.0, self.profile.observation_noise, size=state.shape)
        return (state + noise).astype(np.float32)

    def sample_photometry(self) -> Tuple[float, float]:
        """为当前档案采样一组 ``(亮度, 对比度)`` 乘子。

        每次**观测**采样一次，而不是每台相机各采一次 —— 否则头戴相机与腕部
        相机会对同一场景给出互相矛盾的照明，视觉模型收到的多视角输入不自洽。

        Returns:
            ``(brightness, contrast)``，均以 1.0 为中心。
        """
        brightness = 1.0 + self.rng.uniform(-self.profile.lighting, self.profile.lighting)
        contrast = 1.0 + self.rng.uniform(-self.profile.texture, self.profile.texture)
        return float(brightness), float(contrast)

    def randomize_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """对机器人观测载荷施加**全部观测侧**随机化。

        覆盖 ``L`` 光照、``T`` 纹理、``C`` 相机、``mu`` 状态噪声。
        光照与相机各采样一次并**在全部相机间复用**（包括传感器噪声），
        保证同一场景的多个视角互相一致。

        图像保持原始分辨率（缩放交给适配器，见 ``arena/adapter.py``）。

        中性档案（例如 ``DR-L1``，所有图像侧幅度为 0）下，载荷**原样返回**：
        连数组对象身份都保持，这样基线仿真是可复现的。

        Args:
            observation: ``{"images": {相机名: 画面}, "state": ndarray, ...}``。

        Returns:
            新的观测字典。若有相机扰动，会额外写入
            ``"camera_randomization"`` 字段记录本次采样值（便于日志与复现）。
        """
        result = dict(observation)  # 浅拷贝，避免污染调用方字典
        photometry = (
            self.sample_photometry()
            if (self.profile.lighting > 0 or self.profile.texture > 0)
            else None
        )
        camera = self.sample_camera()

        if photometry is None and camera.is_identity:
            # 无事可做：保留调用方的数组（及其对象身份）
            result["images"] = dict(observation.get("images", {}))
        else:
            # 只采样一次传感器噪声，所有相机共享，保证多视角一致
            images = observation.get("images", {})
            shared_noise = None
            if camera.noise_sigma > 0 and images:
                shape = np.asarray(next(iter(images.values()))).shape
                shared_noise = self.rng.normal(0.0, camera.noise_sigma, size=shape)
            result["images"] = {
                key: self.randomize_camera(self.randomize_image(img, photometry), camera, shared_noise)
                for key, img in images.items()
            }
        if "state" in observation:
            result["state"] = self.randomize_state(observation["state"])
        if not camera.is_identity:
            result["camera_randomization"] = camera.describe()
        return result

    # -- 动力学侧随机化 -----------------------------------------------------
    def randomize_dynamics(self) -> DynamicsRandomization:
        """为当前等级采样一组物理参数缩放系数（因子 ``M``）。

        ``profile.dynamics`` 是最大相对偏差；四个系数**独立采样**并裁剪，
        因此永远不会变成非物理值（负质量/负摩擦）。1 级（或任何
        ``dynamics <= 0`` 的档案）返回恒等采样。

        注意: 本方法只**产生**参数，不施加。施加方式见
            :class:`DynamicsRandomization`。

        Returns:
            采样得到的 :class:`DynamicsRandomization`。
        """
        magnitude = min(max(0.0, float(self.profile.dynamics)), MAX_DYNAMICS_MAGNITUDE)
        if magnitude <= 0.0:
            return DynamicsRandomization()

        def scale() -> float:
            """采样单个缩放系数并裁剪到物理合理区间。"""
            value = 1.0 + self.rng.uniform(-magnitude, magnitude)
            return float(np.clip(value, MIN_DYNAMICS_SCALE, MAX_DYNAMICS_SCALE))

        return DynamicsRandomization(
            mass_scale=scale(),
            friction_scale=scale(),
            actuator_gain_scale=scale(),
            damping_scale=scale(),
        )

    # -- 动作侧随机化 -------------------------------------------------------
    def delay_action(self, action: np.ndarray) -> np.ndarray:
        """对动作施加固定步数延迟，模拟通信/推理延迟（因子 ``N``）。

        实现是一个 FIFO 缓冲：先入队，缓冲区长度不足 ``action_delay_steps`` 时
        一直返回队首元素（"冻结"），之后返回队首并弹出，从而整体滞后 N 步。

        Args:
            action: 单个动作向量。

        Returns:
            延迟后的动作；``action_delay_steps <= 0`` 时原样转换返回。
        """
        if self.profile.action_delay_steps <= 0:
            return np.asarray(action, dtype=np.float32)
        self._delay_buffer.append(np.asarray(action, dtype=np.float32))
        if len(self._delay_buffer) <= self.profile.action_delay_steps:
            # 缓冲尚未填满：保持输出第一个动作，形成"等待"效果
            return self._delay_buffer[0]
        return self._delay_buffer.pop(0)


# ---------------------------------------------------------------------------
# 课程管理器
# ---------------------------------------------------------------------------


class Sim2RealCurriculum:
    """管理四个迁移等级之间的推进。

    这是上层（控制回路 / 集成脚本）实际使用的门面类，内部持有一个
    :class:`Randomizer`，并在切换等级时重建它。
    """

    def __init__(self, config: Optional[Sim2RealConfig] = None) -> None:
        """初始化。

        Args:
            config: sim-to-real 配置；``level`` 会被裁剪到 1..4，``seed`` 传给
                随机化器。``None`` 时用默认配置（1 级）。
        """
        self.config = config or Sim2RealConfig()
        self.level = int(np.clip(self.config.level, 1, 4))
        self.randomizer = Randomizer(CURRICULUM[self.level], seed=self.config.seed)

    @property
    def profile(self) -> DomainRandomization:
        """当前等级的随机化档案。"""
        return self.randomizer.profile

    def advance(self) -> DomainRandomization:
        """推进到下一等级（4 级饱和）。

        **会保留动作延迟缓冲**：切换等级不应该清空"已经发生的历史延迟"，
        否则延迟行为在升级瞬间出现不连续。

        Returns:
            新等级的随机化档案。
        """
        self.level = min(self.level + 1, 4)
        state = self.randomizer._delay_buffer
        self.randomizer = Randomizer(CURRICULUM[self.level], seed=self.config.seed)
        self.randomizer._delay_buffer = state
        return self.profile

    def reset(self, level: Optional[int] = None) -> DomainRandomization:
        """重置到指定等级（默认回到配置里的等级）。

        与 :meth:`advance` 不同，本方法**不保留**延迟缓冲（视为全新开始）。

        Args:
            level: 目标等级；``None`` 时用 ``config.level``。

        Returns:
            重置后等级的随机化档案。
        """
        self.level = int(np.clip(level if level is not None else self.config.level, 1, 4))
        self.randomizer = Randomizer(CURRICULUM[self.level], seed=self.config.seed)
        return self.profile

    def randomize_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """对观测施加光照 / 纹理 / 相机 / 状态噪声随机化。"""
        return self.randomizer.randomize_observation(observation)

    def randomize_dynamics(self) -> DynamicsRandomization:
        """采样应由仿真器施加的物理参数缩放系数。

        随机化器与引擎解耦，因此无法自己写入参数。建议**每个 episode 调用一次**，
        把结果交给环境（Isaac Lab 施加示例见
        :class:`DynamicsRandomization`）。

        Returns:
            采样得到的 :class:`DynamicsRandomization`。
        """
        return self.randomizer.randomize_dynamics()

    def delay_action(self, action: np.ndarray) -> np.ndarray:
        """对动作施加当前等级的延迟。"""
        return self.randomizer.delay_action(action)
