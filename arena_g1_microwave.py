#!/usr/bin/env python3
# 麻雀虽小智能科技（武汉）有限公司
"""
══════════════════════════════════════════════════════════════════════════
  ARENA 0.1.0 — G1-D 双臂人形机器人 | 厨房微波炉操作 | 仿真实验
══════════════════════════════════════════════════════════════════════════

基于 Isaac Lab Arena 0.1.0 的三个核心模块编程实现：
  ╔══════════╦════════════════════════════════╗
  ║ Scene    ║ 厨房场景 + 微波炉 + 机器人位姿 ║
  ║ Embodiment ║ GR1T2 双臂人形 + PINK IK 控制器 ║
  ║ Task     ║ OpenDoorTask (开门度阈值 80%)    ║
  ╚══════════╩════════════════════════════════╝

运行方法与详细步骤
══════════════════════════════════════════════════════════════════════════

【环境要求】
  - Conda 环境: env_isaaclab (Python 3.11 + Isaac Lab 0.47.2)
  - GPU: NVIDIA RTX 4090 (23GB VRAM)，需要 ≥8GB 空闲
  - 显示器: 需要 X11 显示 (DISPLAY 环境变量已设置)
  - 网络: 需要 internet 下载厨房背景 USD 文件（首次运行时）

【步骤 1: 确认 GPU 内存充足】
  $ nvidia-smi
  确保 "Memory-Usage" 至少剩余 8GB。如果有僵尸 Isaac Sim 进程占用，
  用 kill -9 <PID> 手动清理。

【步骤 2: 激活 Conda 环境】
  $ conda activate env_isaaclab
  $ cd /home/rq/文档/project/arena_0_1_0

【步骤 3: 启动 GUI 模式仿真（Mock 后端，不需要 GPU 推理服务器）】
  $ python arena_g1_microwave.py --no_headless --backend mock --max_steps 50

  预期效果：
  - 弹出 Isaac Sim 渲染窗口 (Kit + Hydra 渲染器)
  - 窗口显示厨房背景 (kitchen_background.usd)
  - GR1T2 双臂人形机器人站在微波炉左侧 (位置 -0.4m)
  - 微波炉放在操作台上 (位置 0.4m)
  - 机器人双臂执行随机动作 (Mock 后端生成 36 维动作)
  - 每一步打印 reward、terminated、truncated 状态
  - 约 50 步或 door openness ≥ 80% 后任务结束

【步骤 4 (可选): 对接 VLA 策略服务器】
  终端 A — 启动仿真 (HTTP 客户端):
  $ python arena_g1_microwave.py --no_headless --backend http --max_steps 200

  终端 B — 启动 VLA 策略服务器:
  $ conda activate unifolm-vla
  $ cd /home/rq/文档/project/arena_0_1_0/unifolm-vla
  $ python deployment/model_server/run_real_eval_server.py \
      --ckpt_path models/UnifoLM-VLA-Base1/checkpoints/pytorch_model.pt \
      --port 8777

【程序启动流程 (6 步时序)】

  第 1 步  AppLauncher        ← 启动 Isaac Sim (Kit + PhysX + 渲染)
           耗时: ~60s
  第 2 步  AssetRegistry      ← 注册所有内置资产 (背景/物体/机器人)
           get_asset_by_name("kitchen")()     → 厨房场景
           get_asset_by_name("microwave")()   → 微波炉 (可开门)
           get_asset_by_name("gr1_pink")()    → GR1T2 机器人
  第 3 步  Scene(assets=[...])  ← 组合场景
  第 4 步  OpenDoorTask(microwave, openness_threshold=0.8) ← 定义任务
           耗时: <1s
  第 5 步  ArenaEnvBuilder     ← 生成 Isaac Lab 配置
           ├─ orchestrate()    协调资产间关系 (碰撞组、传感器)
           ├─ compose_manager_cfg()  生成 ManagerBasedRLEnvCfg
           └─ 修复: 移除 last_action 观测 (避免 0 维张量错误)
           耗时: ~1s
  第 6 步  ManagerBasedRLEnv   ← 创建仿真环境
           ├─ Scene → 克隆到 GPU Fabric
           ├─ Observation Manager (8 个 policy 观测项)
           ├─ Action Manager (PINK IK, 36 维动作)
           ├─ Event Manager (reset 逻辑)
           └─ Termination Manager (timeout + door_openness)
           耗时: ~8s

  总启动时间约 70-90 秒 (首次运行含 USD 资产下载)

【动作空间说明】
  36 维动作 = 14 维双臂 IK 目标 + 22 维双手手指关节
  双臂 IK: left/right shoulder_pitch/roll/yaw + elbow_pitch + wrist_yaw/roll/pitch
  手部:   L/R index/middle/pinky/ring/thumb proximal + intermediate + distal

【观测空间说明】
  robot_joint_pos      — 所有关节位置 (变长)
  robot_root_pos       — 机器人世界坐标 (3)
  robot_root_rot       — 机器人世界旋转 (4)
  robot_links_state    — 关键连杆状态 (变长)
  left_eef_pos/quat    — 左末端执行器位姿 (3+4)
  right_eef_pos/quat   — 右末端执行器位姿 (3+4)
  hand_joint_state     — 双手手指关节
  head_joint_state     — 头部关节 (3)
  robot_pov_cam_rgb    — 头戴相机 RGB (512×512×3, 仅 headless 模式)

【已知问题与修复】
  1. 0 维 last_action 错误 → set cfg.observations.policy.actions = None
  2. num_envs 默认 4096     → set cfg.scene.num_envs = 1
  3. 相机渲染已禁用 (Isaac Sim 5.1 camera data 0 dim workaround)
     → GUI 模式禁用相机 enable_cameras=False
  4. Isaac Lab 返回嵌套观测 {"policy": {...}}
     → 直接提取 robot_joint_pos 键
  5. 僵尸进程占用 VRAM       → 启动前 kill -9 清理

══════════════════════════════════════════════════════════════════════════
【本脚本在整体架构中的位置】
══════════════════════════════════════════════════════════════════════════
  本脚本是 arena_vla_1.0 工程中"双臂人形微波炉开门"任务的端到端演示入口，
  位于数据流最上游，扮演"仿真环境侧驱动器"的角色：

      Isaac Sim (Kit + PhysX + USD)
        └─ Arena SDK: Scene / Embodiment / Task
             └─ ArenaEnvBuilder → ManagerBasedRLEnvCfg → ManagerBasedRLEnv
                  └─ G1MicrowaveRobot   ← 本脚本实现的 RobotInterface 适配器
                       │ 产出 ARENA 规范观测 {"images": {...}, "state": ndarray}
                       ▼
                  arena.adapter.EmbodimentAdapter   (观测编码 / 动作解码)
                       ▼
                  arena.client.ControlLoop ──► InProcessClient (mock 后端)
                                            └─► PolicyClient (HTTP POST /act)
                       ▼
                  UnifoLM-VLA 策略服务器 (unifolm-vla/deployment/model_server)
                       │ 返回 ActionChunk (H 步动作块)
                       ▼
                  robot.execute(action) → env.step() → 物理步进 → 下一轮观测

  关键定位：本脚本既不训练也不做神经网络推理，只做"翻译 + 驱动"。它把 Isaac Lab
  的仿真状态翻译成策略服务器可消费的观测，再把策略输出的动作写回仿真。策略权重
  与推理实现全部位于 unifolm-vla 侧，二者通过 HTTP 协议解耦，因此仿真与策略可以
  分别运行在不同的 conda 环境（env_isaaclab / unifolm-vla）中，互不污染依赖。

【依赖的外部系统】
  1. isaaclab / isaaclab_arena (Isaac Lab 0.47.2 + Arena 0.1.0)
     - isaaclab.app.AppLauncher              : Kit 应用生命周期管理
     - isaaclab_arena.assets.AssetRegistry   : 内置资产注册表
                                               (kitchen / microwave / gr1_pink)
     - isaaclab_arena.scene.scene.Scene      : 资产组合 → SceneCfg
     - isaaclab_arena.tasks.open_door_task   : OpenDoorTask 任务与终止条件
     - isaaclab_arena.environments.*         : ArenaEnvBuilder /
                                               IsaacLabArenaEnvironment
     - isaaclab.envs.ManagerBasedRLEnv       : 管理器式(观测/动作/事件/终止)RL 环境
  2. arena 包 (工程根目录 arena/): adapter / client / config / backends / types
     - 定义 RobotInterface 协议、观测-动作规范化、ControlLoop 闭环与策略后端
  3. UnifoLM-VLA 策略服务器 (仅 --backend http 时需要)
     - 默认 http://127.0.0.1:8777/act，请求/响应载荷经 json_numpy 序列化
     - 因此客户端必须先 json_numpy.patch()，否则 numpy 数组无法被 json 编码
  4. Omniverse 资产 CDN: 首次运行需联网下载 kitchen_background.usd / 微波炉 USD
  5. NVIDIA GPU (RTX 4090, ≥8GB 空闲 VRAM) + CUDA 驱动
  6. X11 显示 (仅 --no_headless GUI 模式需要可用的 DISPLAY)

【运行前提 (缺一不可)】
  - 已 conda activate env_isaaclab (Python 3.11)
  - 无残留 isaac-sim 僵尸进程；否则 VRAM 不足会让 PhysX / Kit 初始化失败
  - --backend http 时策略服务器已在 8777 端口就绪，且 ckpt 已下载完毕
  - --no_headless 时 DISPLAY 可用
  - 相机渲染被强制关闭 (enable_cameras=False)，故本脚本取不到真实相机图像，
    观测中的 "head" 实际是全零占位图 (见上文"已知问题与修复"第 3 条)

【退出与清理语义】
  仿真循环无论成功或抛异常，最终都会走 finally: env.close() → sim_app.close()。
  由于创建环境后把 env.unwrapped.sim._app_control_on_stop_handle 置为 None，
  Kit 不再持有"应用停止即回调关窗"的句柄，关闭时序完全由本脚本显式控制，
  可避免退出阶段 Kit 二次回调造成的崩溃或卡死。
══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse  # 命令行参数解析
import json      # 结果序列化
import logging   # 日志输出
from typing import Any, Dict, Optional

import numpy as np   # 数组运算
import torch          # GPU 张量操作

# 配置日志格式: 时间 [级别] 消息
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 机器人包装器 — 将 Isaac Lab 仿真环境适配为 ARENA RobotInterface
# ═══════════════════════════════════════════════════════════════════

class G1MicrowaveRobot:
    """
    G1-D 微波炉操作机器人的 ARENA 包装器。

    核心职责:
      - 将 CUDA 张量转换为 numpy 数组 (ARENA 框架全部使用 numpy)
      - 将 Isaac Lab 嵌套观测格式映射为 ARENA 规范格式
        {images: {"head": ndarray, "wrist": ndarray}, state: ndarray}
      - 动作维度补齐 (填充/截断到 action_dim=36)
      - 检测 episode 终止 (terminated/truncated/success 标记)

    使用示例:
        robot = G1MicrowaveRobot(env, instruction="open the microwave door", max_steps=100)
        obs = robot.reset()          # 重置环境，获取初始观测
        while robot.is_running():
            action = get_action()     # 从 VLA 策略获取动作
            robot.execute(action)     # 执行动作

    设计要点与取舍:
      - 本类只依赖 Gym/Gymnasium 风格的 env 接口 (reset/step/action_space)，
        不 import 任何 Isaac Lab 模块，因此可以在没有 Isaac Sim 的机器上
        用假的 env 对象做单元测试。
      - 它同时适配两套"方言"：ArenaSimRobot 需要的 RobotInterface
        (reset/get_observation/execute/is_running)，以及 VLA 服务端需要的
        {"images", "state"} 观测字典。
      - 失败语义：reset() 之前调用 get_observation() 会抛 RuntimeError；
        动作维度不匹配则被静默补齐(补零)/截断而非报错 —— 这是刻意的取舍，
        因为策略侧输出维度会随模型版本变化，宁可退化也不中断长时仿真。
      - 本类不做任何图像 resize / 归一化，那属于 EmbodimentAdapter 的职责，
        以此保持"环境适配"与"模型预处理"的边界清晰。
    """

    # ── 观测键名映射 ──────────────────────────────────────────────
    # 按优先级顺序搜索这些键，找到第一个存在的即使用
    FULL_IMAGE_KEYS = (          # 第三人称视图 (头戴相机)
        "agentview_image", "robot_pov_cam_rgb", "full_image", "image", "rgb", "head_image"
    )
    WRIST_IMAGE_KEYS = (         # 腕部相机 (眼在手上)
        "wrist_image", "robot0_eye_in_hand_image", "wrist"
    )
    STATE_KEYS = (               # 本体感知状态
        "joint_pos", "state", "robot_state", "proprio", "policy"
    )
    # 为什么用"候选列表 + 按序命中第一个"而不是写死键名？
    #   Isaac Lab 各任务/各版本的观测键名并不统一：直接式(direct)任务常用扁平键；
    #   管理器式(manager-based)任务把观测嵌在 "policy" 之下；robomimic 风格复刻
    #   环境又用 agentview_image / robot0_eye_in_hand_image 这类键。写死单个键会
    #   让脚本换一个环境就 KeyError，因此这里按"最具体 → 最通用"的优先级逐个
    #   探测，命中即停止。具体键名(如 robot_pov_cam_rgb)可信度高于泛化键名
    #   (如 image / rgb)，放前面可避免误取到语义不同的图像；全部落空时返回
    #   None，由调用方决定退化为占位图还是空状态。
    #   注：本文件 get_observation() 对状态走 robot_joint_pos 直取路径，
    #   STATE_KEYS 目前是给其它观测布局预留的兼容表(保留而非删除)。

    def __init__(self, env, instruction: str = "", max_steps: int = 300):
        """
        初始化机器人包装器。

        参数:
            env: Isaac Lab ManagerBasedRLEnv 实例 (gymnasium 兼容)
            instruction: 自然语言任务描述，例如 "open the microwave door"
            max_steps: 最大步数，超过后自动结束 episode

        异常/副作用:
            构造阶段不会访问仿真资源(不 reset、不 step)，唯一可能失败的是
            _get_action_dim()，但它内部已 try/except 兜底，故构造函数不抛异常。

        实例属性:
            env / instruction / max_steps : 外部注入的运行参数
            steps       : 已执行步数计数器，reset() 清零，execute() 自增
            _last_obs   : 最近一次解包后的观测字典；reset() 前为 None，
                          是 get_observation()/task_finished() 的唯一数据源
            _action_dim : 从 env.action_space 推导的动作维度(GR1T2 为 36)
        """
        self.env = env                         # Isaac Lab 仿真环境
        self.instruction = instruction          # 任务指令文本
        self.max_steps = max_steps              # episode 步数上限
        self.steps = 0                          # 当前已执行步数
        self._last_obs: Optional[Dict[str, Any]] = None  # 缓存的上一次观测
        self._action_dim = self._get_action_dim()        # 动作空间维度

    def _get_action_dim(self) -> int:
        """
        获取动作空间维度。

        GR1T2 微波炉环境: action_dim = 36
          = 14 维双臂 IK 目标 (left/right × 7) + 22 维手部关节

        返回:
            int: 动作向量长度；取 action_space.shape 的最后一维，
                 以兼容 (D,) 与 (N, D) 两种形状的 action_space。

        失败会怎样:
            action_space 缺失、为 None 或形状异常时，吞掉异常并回退 23
            (经典 Franka/单臂 7 维 + 双手类环境的常见值)。宁可给一个可能
            偏小的维度，也不让包装器构造失败 —— 后续 execute() 会再做
            补齐/截断，最坏情况只是动作被截短。
        """
        try:
            return int(self.env.action_space.shape[-1])
        except Exception:
            return 23  # 默认回退值

    # ── RobotInterface 接口实现 ───────────────────────────────────

    def reset(self) -> Dict[str, Any]:
        """
        重置环境并返回初始观测。

        Isaac Lab reset() 返回: (obs_dict, info_dict)
        我们解包后将原始观测缓存到 self._last_obs

        返回格式: {"images": {"head": ndarray, ...}, "state": ndarray}

        实现说明:
            这里刻意不把 env.reset() 的返回值直接往上抛，而是统一经
            _unwrap() 压成扁平 dict 再缓存。原因是 reset() 的返回结构
            在 gym 0.21 / gymnasium 0.29 / Isaac Lab 之间不一致，若把
            差异泄漏给上层 ControlLoop，会让闭环代码到处写兼容分支。
        """
        result = self.env.reset()            # 调用 Isaac Lab 的 reset
        self._last_obs = self._unwrap(result)  # 解包统一格式
        self.steps = 0                        # 重置步数计数器
        return self.get_observation()         # 转换为 ARENA 格式返回

    def get_observation(self) -> Dict[str, Any]:
        """
        从缓存的原始观测提取 ARENA 规范格式。

        ARENA 观测格式 (与 UnifoLM-VLA 模型兼容):
          {
            "images": {
              "head":  ndarray(H, W, 3) uint8,  # 第三人称图像
              "wrist": ndarray(H, W, 3) uint8,   # 腕部相机图像
            },
            "state": ndarray(D,) float32          # 本体感知向量
          }

        注意: GUI 模式下 camera_keys 为空，因此 images 不会包含相机数据。
        此时返回全零占位图像 (224×224×3)。

        返回:
            Dict[str, Any]: {"images": {相机名: uint8 ndarray}, "state": float32 (D,)}
                            state 一定是 1 维 float32；取不到时长度为 0。

        失败会怎样:
            - 未先 reset() → RuntimeError (提前失败优于拿到脏数据)。
            - 某个候选图像键存在但无法转 numpy → _to_numpy() 内部兜底，
              不抛异常；若最终 images 为空则注入占位图，保证下游 payload
              结构完整 (VLA 服务端要求至少有一张 full_image)。
        """
        if self._last_obs is None:
            raise RuntimeError("请先调用 reset() 再调用 get_observation()")

        # ── 提取图像 ──────────────────────────────────────────────
        # 语义约定: "head" 一律指第三人称/头部视角，"wrist" 指手腕眼在手上视角。
        # 这里只做"改名 + 转 numpy"，不做 resize/归一化 —— 键名探测放在环境侧，
        # 是因为只有包装器知道底层环境到底吐了哪些键；而尺寸/数值域的规范化
        # 属于模型侧契约，交给 EmbodimentAdapter，避免两处各自为政。
        images: Dict[str, np.ndarray] = {}
        # 搜索第三人称视图 (按 FULL_IMAGE_KEYS 的优先级顺序，命中即 break，
        # 避免后面的泛化键覆盖掉前面更具体的键)
        for key in self.FULL_IMAGE_KEYS:
            val = self._last_obs.get(key)
            if val is not None:
                images["head"] = self._to_numpy(val)
                break
        # 搜索腕部视图
        for key in self.WRIST_IMAGE_KEYS:
            val = self._last_obs.get(key)
            if val is not None:
                images["wrist"] = self._to_numpy(val)
                break
        # 如果没有任何图像 (GUI 模式)，返回占位图像
        # 为什么要造占位图而不是报错/跳过？
        #   1) 本脚本强制 enable_cameras=False (Isaac Sim 5.1 相机 0 维崩溃规避)，
        #      且 GUI 模式下 camera_keys 为空，所以"没有任何图像"是预期常态而非异常；
        #   2) 下游 EmbodimentAdapter / VLA 服务端把图像列为必填字段，缺失会直接
        #      抛 KeyError 或形状错误，使"无相机"这一可预期情况变成崩溃；
        #   3) 224×224 是 UnifoLM-VLA 的图像输入尺寸，用全零而非随机噪声可以保证
        #      结果是确定性的、可复现的，同时明确表达"此处无真实视觉信息"。
        #   代价是策略在无相机时看到的是恒定黑图，只能依赖 state 分支 —— 这是
        #   当前相机 workaround 下不得不接受的退化。
        if not images:
            # 224×224 与 UnifoLM-VLA 模型输入尺寸一致
            images["head"] = np.zeros((224, 224, 3), dtype=np.uint8)

        # ── 提取本体感知状态 ──────────────────────────────────────
        # 为什么不走通用的 STATE_KEYS 搜索？
        #   本体感知是策略最关键的输入，键名必须确定，不能靠"碰运气命中"。
        #   robot_joint_pos 是管理器式观测里关节位置的标准键，直接取它比遍历
        #   候选表更可控；只有在缺失时才退回嵌套 "policy" 字典再找一次。
        # Isaac Lab 返回嵌套观测: {"policy": {"robot_joint_pos": tensor, ...}}
        # 优先提取 robot_joint_pos (关节位置，最核心的本体感知信息)
        state = self._last_obs.get("robot_joint_pos")
        if state is None:
            # 兼容嵌套格式
            policy = self._last_obs.get("policy")
            if isinstance(policy, dict):
                state = policy.get("robot_joint_pos")
        if state is not None:
            state = self._to_numpy(state)
        else:
            # 没有状态信息时返回空数组
            state = np.zeros(0, dtype=np.float32)

        return {"images": images, "state": np.asarray(state, dtype=np.float32).reshape(-1)}

    def execute(self, action: np.ndarray) -> None:
        """
        执行单个动作。

        参数:
            action: 动作向量 (D,) — 在 GR1T2 环境中 D=36

        处理流程:
          1. 补齐/截断到 action_dim
          2. 裁剪到动作空间限位 (action_space.low / high)
          3. 转换为 torch 张量 → 移动到 GPU (CUDA)
          4. 调用 env.step() 执行一步仿真
          5. 更新缓存的观测 → 步数 +1

        返回:
            None。结果不通过返回值传出，而是写回 self._last_obs 并自增 self.steps，
            由紧接着的 is_running()/task_finished()/get_observation() 读取。

        副作用与失败语义:
            - 每调用一次就真实推进一步物理仿真，不可撤销；不要用它做"试探"。
            - env.step() 抛异常时本方法不做 try/except，异常向上冒泡到 ControlLoop
              的 finally，最终仍会执行 env.close()/sim_app.close() 完成清理。
            - 返回的观测按 _unwrap() 的规则压平后覆盖式写入 _last_obs，不做合并，
              因此上一步的 terminated/success 标记不会"粘住"下一步。
        """
        # ── 1. 动作处理 ───────────────────────────────────────────
        # reshape(-1) 先把 (1, D)/(D,)/(D,1) 等各种形状压成 1 维，保证后面
        # 的长度比较与 np.clip 的逐元素语义成立。
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        # 补齐不足的部分：补 0 而不是补随机值/NaN。0 在归一化动作空间里对应
        # "不动作/中间位"，既不会踢出非法目标，也不会污染后面的限位裁剪。
        if len(arr) < self._action_dim:
            arr = np.pad(arr, (0, self._action_dim - len(arr)))
        # 截断超出的部分：策略头部维度大于环境时只取前 action_dim 维，
        # 顺序上先对齐维度、再裁剪限位，二者不可交换 —— 若先裁剪后补齐，
        # 补齐的 0 会落在裁剪之后从而绕过限位保护，且维度不一致时
        # low/high 的广播可能静默给出错误结果。
        elif len(arr) > self._action_dim:
            arr = arr[: self._action_dim]

        # ── 2. 关节限位裁剪 ───────────────────────────────────────
        # 确保动作值在机器人可执行范围内
        # 目的：VLA 输出的归一化动作经反归一化后可能越界，直接送进 action manager
        # 会让 IK 目标飞到工作空间外(触发求解失败或关节突跳)。裁剪是最后一道
        # 安全网。这里按 min(len) 取交集长度，是为了兼容 action_space 只声明了
        # 部分自由度(如只控双臂、不含手指)的情况。
        if hasattr(self.env, "action_space") and self.env.action_space is not None:
            low = np.asarray(self.env.action_space.low).reshape(-1)
            high = np.asarray(self.env.action_space.high).reshape(-1)
            m = min(len(arr), len(low))
            arr[:m] = np.clip(arr[:m], low[:m], high[:m])

        # ── 3. 转换为 GPU 张量 ─────────────────────────────────────
        # unsqueeze(0) 补出 batch 维：Isaac Lab 的动作张量形状固定为 (num_envs, D)，
        # 本脚本 num_envs=1，所以是 (1, 36)。设备跟随 env.unwrapped.device，
        # 避免 CPU/GPU 张量混用导致的隐式拷贝或报错。
        action_tensor = torch.from_numpy(arr).float().unsqueeze(0)  # (1, 36)
        if hasattr(self.env.unwrapped, "device"):
            action_tensor = action_tensor.to(self.env.unwrapped.device)

        # ── 4. 执行仿真步 ─────────────────────────────────────────
        # inference_mode 下不做 autograd 记录：闭环推理不需要梯度，关掉可省显存
        # 与少量开销；同时也能避免长时循环里意外累积计算图。
        with torch.inference_mode():  # 关闭梯度计算，节省内存
            result = self.env.step(action_tensor)

        # ── 5. 缓存结果 ───────────────────────────────────────────
        # 步数仅在这里自增，因此 steps 精确等于"已执行的 env.step() 次数"，
        # 这也是 is_running() 能做硬上限保护的前提。
        self._last_obs = self._unwrap(result)
        self.steps += 1

    def is_running(self) -> bool:
        """
        判断 episode 是否应该继续。

        返回 False 的情况:
          - 步数超过上限 (self.max_steps)
          - 任务已完成 (task_finished() 返回 True)

        注意:
            这里用 and 短路，max_steps 与任务终止是"或"关系 —— 任一条件成立
            即停止。max_steps 是仿真侧的安全阀：即使任务永远不成功(例如策略
            始终输出噪声)，也不会无限跑下去占着 GPU。
        """
        return self.steps < self.max_steps and not self.task_finished()

    def task_finished(self) -> bool:
        """
        判断任务是否已完成。

        成功条件 (OpenDoorTask):
          微波炉门打开度 ≥ 80% (openness_threshold=0.8)
          → terminated["success"] = True

        检查的标记键: "terminated", "truncated", "done", "success"

        实现细节:
            - 用 np.asarray(v).any() 而非 bool(v)：v 往往是形状 (1,) 的
              torch/GPU 张量，直接 bool() 在多元素时抛异常、在 GPU 张量上
              也会报错，先转 numpy 再 any() 最稳。
            - "truncated" 也被视为结束：它代表达到时间上限而被截断，
              对闭环而言同样必须退出，否则会继续往已自动 reset 的环境里灌动作。
            - 这里无法区分"成功终止"与"超时截断"，两者都返回 True；
              真实成功率由 ControlLoop 汇总的信息决定。
        """
        if self._last_obs is None:
            return False
        for key in ("terminated", "truncated", "done", "success"):
            v = self._last_obs.get(key)
            if v is not None and bool(np.asarray(v).any()):
                return True
        return False

    # ── 内部工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _to_numpy(val):
        """
        将任意类型转换为 numpy 数组。

        支持: numpy 数组 / torch 张量 (CPU/GPU) / 标量 / 列表

        参数:
            val: 任意待转换对象。

        返回:
            np.ndarray；已是 numpy 时原样返回(不拷贝，避免每步多一次大数组复制)。

        失败会怎样:
            前两个分支不会失败；np.asarray 抛异常时用 float32 再试一次，
            若仍失败则异常向上冒泡(此处的 except 只是把失败点后移，并非吞掉)。
            之所以对 torch 显式走 .cpu()，是因为 GPU 张量无法被 np.asarray 直接
            消费，必须先搬到主机内存。
        """
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, (torch.Tensor,)):
            return val.cpu().numpy()  # GPU 张量 → CPU → numpy
        try:
            return np.asarray(val)
        except Exception:
            return np.asarray(val, dtype=np.float32)

    @staticmethod
    def _unwrap(result) -> Dict[str, Any]:
        """
        统一 Isaac Lab env.step() / env.reset() 的各种返回格式。

        Isaac Lab 返回格式 (gymnasium 风格):
          5 元组: (obs, reward, terminated, truncated, info)
          4 元组: (obs, reward, done, info)
          2 元组: (obs, info)

        所有格式统一转换为扁平字典，方便后续处理。

        参数:
            result: env.reset() / env.step() 的原始返回值。

        返回:
            Dict[str, Any]: 观测键与 terminated/truncated/done/reward 标记
            混在同一个字典里。这样 get_observation() 与 task_finished() 都只需
            面对一种结构，把"版本差异"集中在唯一一处处理。

        失败会怎样:
            不抛异常。无法识别的类型返回空 dict {}，表现为"这一步没有观测"，
            由上层自行退化(状态为空数组/占位图)，而不是中断仿真。
        """
        if isinstance(result, tuple):
            if len(result) == 5:
                # 5 元组格式 (gymnasium 0.29+)
                obs, reward, terminated, truncated, info = result
                merged = dict(obs) if isinstance(obs, dict) else {}
                merged["terminated"] = G1MicrowaveRobot._to_numpy(terminated)
                merged["truncated"] = G1MicrowaveRobot._to_numpy(truncated)
                merged["reward"] = G1MicrowaveRobot._to_numpy(reward)
                return merged
            if len(result) == 4:
                # 4 元组格式 (经典 gym / 旧版 gymnasium)
                obs, reward, done, info = result
                merged = dict(obs) if isinstance(obs, dict) else {}
                merged["done"] = G1MicrowaveRobot._to_numpy(done)
                merged["reward"] = G1MicrowaveRobot._to_numpy(reward)
                return merged
            # 2 元组格式 (老版本)
            obs, info = result[0], result[1] if len(result) > 1 else {}
            merged = dict(obs) if isinstance(obs, dict) else {}
            if isinstance(info, dict):
                for k, v in info.items():
                    merged[k] = G1MicrowaveRobot._to_numpy(v) if isinstance(v, (torch.Tensor,)) else v
            return merged
        if isinstance(result, dict):
            return result
        return {}  # 回退


# ═══════════════════════════════════════════════════════════════════
# 主程序 — 基于 Arena SDK 三核心模块 (Scene, Embodiment, Task)
# 【分节 0/9 · 入口 main() 总览】下面 main() 内的分节顺序即启动时序:
#   命令行参数 → 启动引擎 → 场景构建 → 任务定义 → 配置生成与修复
#   → 环境创建 → 适配器与客户端 → 闭环执行 → 清理
# ═══════════════════════════════════════════════════════════════════

def main():
    """
    Arena 0.1.0 仿真实验入口。

    执行流程:
      ① 启动 Isaac Sim          → AppLauncher
      ② 构建场景 (Scene)        → Kitchen + Microwave + GR1T2
      ③ 定义任务 (Task)          → OpenDoorTask
      ④ 组合环境                 → IsaacLabArenaEnvironment
      ⑤ 生成配置                 → ArenaEnvBuilder → ManagerBasedRLEnvCfg
      ⑥ 创建仿真环境             → ManagerBasedRLEnv(cfg=cfg)
      ⑦ 包装为 ARENA 接口       → G1MicrowaveRobot
      ⑧ 闭环执行                 → ControlLoop.run()

    设计约束(写在前面，便于理解后文的顺序安排):
      - main() 内部大量使用"函数内 import"(如 isaaclab.app / isaaclab_arena)。
        这不是风格问题而是硬性要求：这些模块必须等 AppLauncher 把 Kit 启动
        完毕之后才可导入，否则 Omniverse 的 USD/PhysX 绑定尚未就绪，导入期
        就会失败。因此本文件顶部只 import 与 Isaac Sim 无关的通用依赖。
      - 全流程无返回值；实验结果由 ControlLoop 打印成 JSON 日志，失败通过
        logger.exception 记录，进程退出码不反映任务成败。
      - 资源释放集中在最后的 finally，任何一步抛异常都会走到那里。
    """

    # ── 命令行参数 ────────────────────────────────────────────────
    # 【分节 1/9 · 命令行参数 / cli-args】
    # 这些开关决定后面所有分支：headless 决定 Kit 是否开窗渲染，
    # backend 决定用进程内 mock 还是 HTTP 策略服务器，
    # max_steps 是仿真侧的硬性步数上限(与任务自身的 5s 超时形成双保险)。
    parser = argparse.ArgumentParser(
        description="ARENA 0.1.0 — G1-D 厨房微波炉仿真实验"
    )
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "http"],
        help='策略后端: "mock"=随机动作(测试用) / "http"=对接VLA服务器',
    )
    parser.add_argument(
        "--server_url",
        default="http://127.0.0.1:8777/act",
        help="VLA 策略服务器地址 (仅 --backend http 时有效)",
    )
    parser.add_argument(
        "--instruction",
        default="open the microwave door",
        help="自然语言任务指令",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=100,
        help="最大仿真步数 (每步 = 4 个物理子步 ≈ 0.02s)",
    )
    # GUI 控制: 默认 headless=True (无渲染窗口)，用 --no_headless 切换
    parser.add_argument("--headless", action="store_true", default=True,
                        help="无头模式 (不显示渲染窗口)")
    parser.add_argument("--no_headless", dest="headless", action="store_false",
                        help="GUI 模式 (显示 Isaac Sim 渲染窗口)")
    args = parser.parse_args()

    # ══════════════════════════════════════════════════════════════
    # 第 1 步: 启动 Isaac Sim 仿真引擎
    # 【分节 2/9 · 启动引擎 / launch-engine】
    # ══════════════════════════════════════════════════════════════
    # AppLauncher 是 Isaac Sim 的入口类，负责:
    #   - 加载 Kit 应用配置 (.kit 文件)
    #   - 初始化 PhysX 物理引擎 (GPU 加速)
    #   - 启动 Hydra 渲染器 (GUI 模式)
    #   - 返回 sim_app 句柄 (用于后续关闭)
    from isaaclab.app import AppLauncher

    # ★ 关键: 完全禁用相机渲染 (Isaac Sim 5.1 camera data 0 dim workaround)
    # 无论 headless 还是 GUI，相机都会触发 syntheticdata 崩溃
    # 为什么把 enable_cameras 硬编码为 False 而不暴露成命令行开关？
    #   因为这不是性能取舍而是崩溃规避：Isaac Sim 5.1 的相机管线在
    #   syntheticdata/Replicator 初始化时返回 0 维的相机数据张量，
    #   后续 ObservationManager 会按 (N,H,W,C) 解析并直接抛维度错误，
    #   而该错误发生在 Kit 内部，脚本侧无法捕获补救。相机一旦参与构建，
    #   整个进程会在启动阶段崩掉，所以这里只能一刀切禁用。
    #   代价：观测里拿不到真实图像(见 get_observation() 的占位图逻辑)。
    enable_cameras = False
    logger.info(
        "▶ 启动 Isaac Sim (headless=%s, cameras=%s)...",
        args.headless, enable_cameras,
    )
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=enable_cameras)
    sim_app = app_launcher.app

    # ══════════════════════════════════════════════════════════════
    # 第 2 步: Scene 模块 — 构建仿真场景
    # 【分节 3/9 · 场景构建 / scene-build】
    # ══════════════════════════════════════════════════════════════
    # Scene 是 Arena SDK 的三大核心模块之一，负责:
    #   - 背景 (kitchen): 厨房环境，包含操作台、橱柜等 3D 模型
    #   - 物体 (microwave): 可开门的微波炉，继承 Openable 接口
    #   - 机器人 (gr1_pink): GR1T2 双臂人形 + PINK IK 逆运动学控制器
    import argparse as _argparse
    from isaaclab_arena.assets.asset_registry import AssetRegistry

    # AssetRegistry 是单例模式，自动发现并注册所有内置资产
    asset_registry = AssetRegistry()

    # ── 2a. Embodiment: 加载 GR1T2 双臂人形机器人 ─────────────────
    # gr1_pink = GR1T2 机器人 + PINK IK 控制器
    #   - 14 自由度双臂: shoulder_pitch/roll/yaw + elbow_pitch + wrist_yaw/roll/pitch × 2
    #   - 22 自由度手部: 双手各 11 个手指关节
    #   - PINK IK: 接收末端执行器 (EEF) 位姿目标 → 计算关节角度
    #   - 重力补偿: 抵消机器人自重，使 IK 控制更稳定
    logger.info("▶ 加载 GR1T2 Pink embodiment (双臂人形 + PINK IK 控制器)...")
    embodiment = asset_registry.get_asset_by_name("gr1_pink")(
        enable_cameras=enable_cameras  # 是否启用头戴相机 (robot_pov_cam)
    )
    # 设置机器人的初始位姿 (站在微波炉左侧 40cm 处，面朝微波炉)
    # 为什么在这里(构建 Scene 之前)就把位姿写进 embodiment，而不是建完场景再改？
    #   Arena 的资产生命周期是"先声明位姿 → 再 orchestrate/克隆到 GPU Fabric"，
    #   位姿会在生成 SceneCfg 时被烘焙进 prim 的 transform。若等 env 建好再改，
    #   修改的是运行时 prim 而非配置，reset 事件(重置位姿)会把它覆盖回去，
    #   表现为"第一帧对了、reset 后机器人漂回原点"这类难查问题。
    embodiment.set_initial_pose(
        pose=_import_pose()(
            position_xyz=(-0.4, 0.0, 0.0),          # 位置: X=-0.4m (左侧)
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),     # 旋转: 无旋转 (面朝 +X)
        )
    )

    # ── 2b. Background: 厨房场景背景 ──────────────────────────────
    # kitchen_background.usd 包含:
    #   - 操作台 (操作面高度 ~0.9m)
    #   - 橱柜、抽屉
    #   - 墙壁、地板
    # USD 文件从 NVIDIA Omniverse CDN 下载 (首次运行时需网络)
    logger.info("▶ 加载厨房背景 (kitchen_background.usd)...")
    background = asset_registry.get_asset_by_name("kitchen")()

    # ── 2c. Object: 微波炉物体 ────────────────────────────────────
    # Microwave 继承 Openable 接口:
    #   - .is_open(threshold)  → 检查门是否打开超过 threshold
    #   - .close(percentage)   → 重置时关闭门到指定百分比
    logger.info("▶ 加载微波炉物体 (Microwave039.usd)...")
    # 注意: 这行 Openable 导入在当前实现里并未被引用(仅作接口文档/预留)；
    # 保留而不删除，避免影响后续按接口类型做 isinstance 分支的扩展。
    from isaaclab_arena.affordances.openable import Openable

    microwave = asset_registry.get_asset_by_name("microwave")()
    # 设置微波炉的初始位姿 (放在操作台上，门朝机器人)
    microwave_pose = _import_pose()(
        position_xyz=(0.4, -0.00586, 0.22773),     # 操作台上方
        rotation_wxyz=(0.7071068, 0, 0, -0.7071068),  # 门朝 -X (朝机器人)
    )
    microwave.set_initial_pose(microwave_pose)

    # ── 2d. 组合场景 ──────────────────────────────────────────────
    # Scene 类负责:
    #   - 管理所有资产 (通过名称索引)
    #   - 生成 scene_cfg → Isaac Lab 的 SceneCfg 配置类
    #   - 自动分配 prim_path: /World/envs/env_0/kitchen, .../microwave, .../Robot
    from isaaclab_arena.scene.scene import Scene

    assets = [background, microwave]  # 注意: 机器人由 embodiment 单独管理
    scene = Scene(assets=assets)

    # ══════════════════════════════════════════════════════════════
    # 第 3 步: Task 模块 — 定义任务逻辑
    # 【分节 4/9 · 任务定义 / task-definition】
    # ══════════════════════════════════════════════════════════════
    # OpenDoorTask 是 Arena SDK 内置的开门任务:
    #   openness_threshold=0.8 → 门打开 80% 即判定成功
    #   reset_openness=0.2     → 重置时把门关到 20% (不完全关，提供多样性)
    #   episode_length_s=5.0   → 最长 5 秒 (对应 ~250 步，每步 0.02s)
    #
    # 内部自动生成:
    #   termination_cfg: time_out + success(door.openness ≥ 0.8)
    #   events_cfg:      reset_all_joints + close_door + reset_pose
    #   metrics:         SuccessRate + DoorMovedRate
    from isaaclab_arena.tasks.open_door_task import OpenDoorTask

    task = OpenDoorTask(
        microwave,                  # 可开门物体 (必须继承 Openable)
        openness_threshold=0.8,     # 开门度 ≥ 80% → 任务成功
        reset_openness=0.2,         # 重置时关门到 20%
        episode_length_s=5.0,       # 5 秒超时 (实际上由 max_steps 控制)
    )
    logger.info("▶ 任务: %s (开门阈值=80%%, 重置门度=20%%)", task.__class__.__name__)

    # ══════════════════════════════════════════════════════════════
    # 第 4 步: 构建 Isaac Lab 仿真配置
    # 【分节 5/9 · 配置生成与修复 / cfg-build-and-patch】
    # ══════════════════════════════════════════════════════════════
    # 注意: 绕过 gymnasium 注册机制 (gym.make/gym.register)
    # 直接使用 ManagerBasedRLEnv(cfg=cfg) 实例化
    # 原因: Arena SDK 的 last_action 观测在首次 reset 时产生 0 维张量,
    #       导致 gym.make() 中 ObservationManager 初始化失败
    #
    # 展开说明"为什么必须绕过注册":
    #   gym.make(id, cfg=cfg) 会先做 env_spec 查表与 wrapper 链包装，其间
    #   gymnasium 会对观测空间做一次 consistency check(sample 一个观测并与
    #   observation_space 比对)。Arena 生成的 cfg 里含 last_action 观测项，
    #   首次 reset 尚未填充该缓冲，于是得到 0 维张量；0 维张量既无法匹配
    #   Box 空间形状，也会让 ObservationManager 在 stack/reshape 时抛
    #   TypeError。也就是说失败发生在 gym 包装层而非我们的代码里，脚本侧
    #   无法用 try/except 兜住。直接调用 ManagerBasedRLEnv(cfg=cfg) 跳过
    #   注册表与 wrapper 链，只保留 Isaac Lab 自身的初始化路径，行为等价
    #   但不会触发那次一致性检查。
    #   代价：不能再用 env_id 字符串复用环境；本脚本是一次性实验入口，
    #   不需要注册表能力，这个取舍是划算的。
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

    # ── 组合场景 + 实施例 + 任务 ───────────────────────────────────
    arena_env = IsaacLabArenaEnvironment(
        name="gr1_open_microwave",   # 环境名称 (用于日志)
        embodiment=embodiment,        # GR1T2 + PINK IK
        scene=scene,                  # 厨房 + 微波炉
        task=task,                    # OpenDoorTask
    )

    # ── 构建 argparse Namespace (ArenaEnvBuilder 从这里读取参数) ──
    arena_argparse_ns = _argparse.Namespace(
        headless=args.headless,
        enable_cameras=False,        # ★ 已禁用 (Isaac Sim 5.1 camera data 0 dim workaround)
        device="cuda:0",             # GPU 设备
        num_envs=1,                  # 并行环境数 = 1
        disable_fabric=True,         # 单环境时禁用 Fabric (GPU 批处理加速)
        seed=42,                     # 随机种子 (保证可复现)
        task=args.instruction,
        embodiment="gr1_pink",
        object=None,
        teleop_device=None,
        mimic=False,                 # 非模仿学习模式
        video=False,
        video_length=0,
        video_interval=0,
    )

    # ── ArenaEnvBuilder: 核心配置生成器 ───────────────────────────
    builder = ArenaEnvBuilder(arena_env, arena_argparse_ns)

    # orchestrate(): 资产间协调
    #   - 碰撞组分配 (kitchen→0, microwave→0, robot→0)
    #   - 激活 contact sensors (microwave 的碰撞检测)
    #   - 注册自定义观测函数 (get_all_robot_link_state, get_eef_*)
    #   - 注册自定义事件 (set_object_pose, microwave.close)
    #   - 注册终止条件 (microwave.is_open)
    builder.orchestrate()

    # compose_manager_cfg(): 生成完整的 Isaac Lab 环境配置
    cfg = builder.compose_manager_cfg()
    # 生成的结构 (IsaacLabArenaManagerBasedRLEnvCfg):
    #   scene_cfg:      kitchen (AssetBaseCfg) + microwave (ArticulationCfg) + robot (ArticulationCfg)
    #   observations:   Policy (8 项) + Camera (robot_pov_cam_rgb)
    #   actions:        PinkInverseKinematicsActionCfg (PINK IK, 36 维)
    #   events:         reset_all + reset_door_state + reset_openable_object_pose
    #   terminations:   time_out + success(door_openness ≥ 0.8)

    # ── 修复 1: 移除 last_action 观测项 ───────────────────────────
    # Arena SDK 的 actions 观测 (last_action) 在初始 reset() 时产生
    # 0 维张量，导致 ObservationManager 报 TypeError
    # 解决的正是哪个问题: 把 policy 组里的 actions 项置 None，等价于"从观测
    # 组中摘除该项"。ObservationManager 只遍历非 None 的项，于是不再尝试
    # 拼接这个尚未初始化的 0 维张量。置 None 而不是 delattr，是为了与
    # Isaac Lab 的 dataclass 配置约定一致(字段必须存在，值可为 None)。
    # 影响: VLA 侧因此拿不到上一帧动作(last_action)这一输入模态，只能靠
    # 图像与本体状态推断 —— 这是为跑通 Isaac Sim 5.1 付出的能力代价。
    if hasattr(cfg.observations.policy, "actions"):
        cfg.observations.policy.actions = None

    # ── 修复 1b: 禁用相机观测 ─────────────────────────────────────
    # Isaac Sim 5.1 camera data 0 dim workaround: 禁用相机观测
    # 与上面的 enable_cameras=False 是同一 workaround 的两道防线：前者在
    # Kit 侧不创建相机传感器，这里在配置侧再把相机观测组摘掉。二者都保留
    # 是因为相机观测可能由 Arena 的 orchestrate() 独立注入，单靠 Kit 侧
    # 开关未必能挡住；双保险可避免任何一条路径把 0 维相机张量带进
    # ObservationManager。
    if hasattr(cfg.observations, "camera_obs"):
        cfg.observations.camera_obs = None

    # ── 修复 2: 覆盖默认并行环境数 ────────────────────────────────
    # Arena SDK 默认 num_envs=4096 (GPU batch)，单环境应设为 1
    # 解决什么问题: 4096 个并行环境是为大规模 RL 训练准备的，若照搬会立刻
    # 申请数十 GB 显存(每个环境都要克隆一整套厨房+微波炉+人形资产)，
    # 在 23GB 的 4090 上必然 OOM；而且所有观测/终止张量都会带 (4096, ...)
    # 的前导维，R^1 的单机推理根本不需要。改成 1 后，动作张量形状固定为
    # (1, 36)，与 G1MicrowaveRobot.execute() 的 unsqueeze(0) 假设一致。
    cfg.scene.num_envs = 1

    # ══════════════════════════════════════════════════════════════
    # 第 5 步: 创建仿真环境
    # 【分节 6/9 · 环境创建 / env-instantiation】
    # ══════════════════════════════════════════════════════════════
    # ManagerBasedRLEnv 是 Isaac Lab 的核心环境类，初始化过程:
    #   ① 解析 scene_cfg → 在 GPU 上用 Fabric 构建场景
    #   ② 初始化 Observation Manager (8 个 policy 观测项)
    #   ③ 初始化 Action Manager (PINK IK, 36 维动作)
    #   ④ 初始化 Event Manager (reset 逻辑)
    #   ⑤ 初始化 Termination Manager (time_out + success)
    env = ManagerBasedRLEnv(cfg=cfg)
    # 禁用 Isaac Sim 的自动关闭回调 (避免关闭时崩溃)
    # 作用: sim 在构造时会把自己的一个回调句柄注册到 Kit 的"应用停止"事件上，
    #   使得窗口被关闭/App 退出时由 Kit 反过来触发 sim 的清理逻辑。而本脚本
    #   的 finally 已经显式执行 env.close() → sim_app.close()，两条路径同时
    #   清理同一批 GPU/PhysX 资源会触发重复释放(use-after-free)，典型表现是
    #   退出阶段段错误或进程挂死。把句柄置 None 后 Kit 侧不再持有该回调，
    #   关闭时序完全由脚本掌控，退出即可干净返回。
    # 注意: 该属性属 Isaac Sim 内部实现细节，故加了 type: ignore。
    env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

    logger.info(
        "✓ G1 Microwave 环境已创建 — action_dim=%d (36维: 14双臂IK + 22手指关节)",
        int(env.action_space.shape[-1]),
    )

    # ══════════════════════════════════════════════════════════════
    # 第 6 步: ARENA 适配器 + 策略客户端
    # 【分节 7/9 · 适配器与客户端 / adapter-and-client】
    # ══════════════════════════════════════════════════════════════
    from arena.adapter import EmbodimentAdapter
    from arena.client import ControlLoop, PolicyClient
    from arena.config import AdapterConfig, ClientConfig

    action_dim = int(env.action_space.shape[-1])
    # EmbodimentAdapter: 负责机器人观测/动作与 VLA 规范之间的双向转换
    #   - 观测: Isaac Lab 格式 → VLA 规范化 Observation
    #     (resize 图像到 224×224, 归一化 proprio 到 [-1, 1])
    #   - 动作: VLA 预测的动作块 → robot-native 命令
    #     (反归一化, 裁剪到关节限位)
    adapter = EmbodimentAdapter(
        AdapterConfig(robot_type="unitree_g1", action_dim=action_dim),
        instruction=args.instruction,
    )
    # 包装仿真环境为 ARENA RobotInterface
    robot = G1MicrowaveRobot(env, instruction=args.instruction, max_steps=args.max_steps)

    # ── 策略客户端 (Mock 后端 / HTTP 后端) ─────────────────────────
    # 两种后端产出同一种接口对象(都只需实现 infer(obs, instruction))，
    # 因此 ControlLoop 完全不必知道策略跑在进程内还是远端。
    if args.backend == "mock":
        # Mock 后端: 生成 36 维随机动作 (在 [-1, 1] 范围均匀分布)
        # 用于测试仿真环境是否正常工作，不依赖 GPU 推理服务器
        from arena.backends import build_backend
        from arena.config import ServerConfig
        from arena.types import ActionChunk

        backend = build_backend(ServerConfig(backend="mock"))
        # 为什么必须手动改 action_dim？
        #   ServerConfig 只指定了 backend="mock"，Mock 后端的 action_dim 走的是
        #   类内默认值 7 (单臂 Franka 风格)。若不覆盖，mock 只会吐 7 维动作，
        #   经 G1MicrowaveRobot.execute() 补齐后变成"前 7 维随机 + 后 29 维恒 0"，
        #   双臂与手指完全不动，看起来像环境坏了，实际是后端维度没对齐。
        #   这里显式对齐到 env.action_space 的真实维度(=36)，mock 才会生成
        #   覆盖全部自由度的随机动作。注意 action_dim 在 mock 推理时才被读取，
        #   因此必须在建立 client(即首次 infer)之前赋值。
        backend.action_dim = action_dim  # 设置动作维度 (否则默认 7)

        # 进程内客户端: 直接调用 backend.infer(), 跳过 HTTP 往返
        class InProcessClient:
            """
            进程内策略客户端 (mock 专用)。

            为什么需要它: ControlLoop 期望 client 具备
            infer(observation, instruction) → ActionChunk 这一接口，
            mock 后端本身只提供 backend.infer(list[dict], instruction)，
            形状不一致。这里用一个极薄的类把它包成与 PolicyClient 同形的
            鸭子类型对象，从而让 ControlLoop 对"策略在进程内还是远端"
            完全无感知 —— 无需在闭环里写 if backend == "mock" 分支。

            额外收益: 不走 HTTP 就绕开了序列化，obs.to_dict() 得到的
            numpy 数组可以直接传给后端，既省往返延迟也便于调试。

            失败会怎样: 不捕获异常，backend.infer 的错误直接冒泡给
            ControlLoop，最终由 main() 的 except 记录。
            """

            def infer(self, obs, instruction=None):
                """
                执行一次 mock 推理。

                参数:
                    obs: ARENA Observation 对象(需实现 to_dict())。
                    instruction: 任务指令文本；为 None 时退化为空串。

                返回:
                    ActionChunk: 内含 backend 生成的 (H, 36) 动作块。

                注意: 后端接口要求批量输入，故这里把单条观测包成
                      [obs.to_dict()] 再取回动作块。
                """
                return ActionChunk(
                    actions=backend.infer([obs.to_dict()], instruction or "")
                )
        client = InProcessClient()
    else:
        # HTTP 后端: 通过 HTTP POST /act 对接 VLA 策略服务器
        # 每步发送观测, 接收动作块 (H 个动作, H=8)
        import json_numpy
        json_numpy.patch()  # 启用 numpy 数组的 JSON 序列化
        client = PolicyClient(ClientConfig(server_url=args.server_url))

    # ══════════════════════════════════════════════════════════════
    # 第 7 步: 闭环执行
    # 【分节 8/9 · 闭环执行 / control-loop】
    # ══════════════════════════════════════════════════════════════
    # ControlLoop 实现标准的 VLA 闭环:
    #   while robot.is_running():
    #       ① robot.get_observation()         → 获取机器人观测
    #       ② adapter.encode_observation()    → 编码为 VLA 格式
    #       ③ client.infer()                  → 调用策略推理
    #       ④ adapter.decode_chunk()          → 解码动作块
    #       ⑤ for action in chunk:            → 逐动作执行
    #            robot.execute(action)
    #
    # 注意这是"动作分块(open-loop chunk)"式闭环而非逐步闭环：一次推理得到
    # H 步动作后连续执行，中途不重新观测。好处是把昂贵的 VLA 前向次数降低
    # H 倍；代价是块内无法纠偏，策略需自行保证动作序列的连贯性。
    # max_episodes=1：本脚本定位为单次演示/冒烟测试，不做多轮统计评估。
    logger.info("▶ 启动闭环控制 (max_steps=%d, backend=%s)...", args.max_steps, args.backend)
    loop = ControlLoop(
        robot, client, adapter,
        instruction=args.instruction,
        max_episodes=1,          # 运行 1 个 episode
    )

    try:
        results = loop.run(args.instruction)
        logger.info("✓ 实验结果: %s", json.dumps(results, indent=2))
    except Exception as e:
        # 宽泛捕获是刻意的：仿真进程一旦异常退出就可能留下占用 VRAM 的僵尸
        # 进程，而 finally 里的清理才是本脚本最不能跳过的一步。这里记全栈轨迹
        # (logger.exception) 便于定位，同时让控制流继续走到 finally。
        logger.exception("✗ 错误: %s", e)
    finally:
        # 【分节 9/9 · 清理 / teardown】
        # 清理资源
        # 顺序不可颠倒：先 env.close() 释放 PhysX/渲染等仿真资源，再
        # sim_app.close() 关闭 Kit 应用本身。反过来的话 env 会在已销毁的
        # Kit 上下文里做清理，触发 USD/Omniverse 绑定层的崩溃。
        # 由于前面把 _app_control_on_stop_handle 置 None，此处是唯一的关闭路径。
        env.close()
        sim_app.close()
        logger.info("■ 仿真关闭完成。")


def _import_pose():
    """
    懒加载 Pose 类。

    Pose 是 isaaclab_arena.utils.pose 中的位姿表示类，
    需要 Isaac Sim 启动后才能导入 (依赖 Omniverse USD 绑定)。

    为什么做成函数而不在文件顶部直接 import？
        isaaclab_arena.utils.pose 在导入时会拉起 Omniverse/USD 的 Python
        绑定，而该绑定只有在 AppLauncher 启动 Kit 之后才存在。若写在文件顶部，
        连 `python arena_g1_microwave.py --help` 都会因导入失败而报错。包装成
        函数后，导入时机被推迟到真正的场景构建阶段(见 main 中的 _import_pose()())。

    参数: 无。
    返回: Pose 类本身(注意返回的是"类"而不是实例，调用处再用
          position_xyz / rotation_wxyz 构造实例)。
    失败会怎样: 若在 AppLauncher 之前调用，import 语句会抛
                ModuleNotFoundError 或 Omniverse 绑定错误 —— 这时应检查
                启动顺序，而不是在本函数内兜底。
    """
    from isaaclab_arena.utils.pose import Pose
    return Pose


# ── 程序入口 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    main()