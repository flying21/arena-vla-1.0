#!/usr/bin/env python3
# 麻雀虽小智能科技（武汉）有限公司
"""ARENA + Isaac Lab VLA inference pipeline demo.

Connects an Isaac Lab simulation environment to a VLA policy server
(UnifoLM-VLA running on http://127.0.0.1:8777/act) through ARENA's
adapter/client framework.

Usage (in env_isaaclab):
    # Quick test with Cartpole + Mock backend (no GPU/VLA needed)
    python arena_isaaclab_demo.py

    # With the running VLA server:
    python arena_isaaclab_demo.py --backend http

    # With a manager-based locomotion task (Unitree Go2):
    python arena_isaaclab_demo.py --env Isaac-Velocity-Flat-Unitree-Go2-v0 --backend mock

═══════════════════════════════════════════════════════════════════════
【本脚本在整体架构中的位置】— 与 arena_g1_microwave.py 是同一层级的"环境侧驱动"
═══════════════════════════════════════════════════════════════════════
  arena_vla_1.0 里有两个平行的仿真入口，职责相同、泛化程度不同：

    arena_g1_microwave.py  : 专用。绑定 Arena SDK 的 Kitchen + Microwave +
                             GR1T2 具体资产，目的是验证真实操作任务。
    本文件                  : 通用。不依赖 Arena SDK，用 gymnasium 的 env id
                             (默认 Isaac-Cartpole-Direct-v0) 拉起任意 Isaac Lab
                             任务，用来快速验证"ARENA 适配层 + 策略客户端 + VLA
                             服务器"这条链路本身是否通畅。

  数据流(两个脚本完全一致，只是环境来源不同)：

      Isaac Lab 环境 (gym.make / ManagerBasedRLEnv)
        └─ IsaacLabRobotWrapper       ← 本文件实现的 RobotInterface 适配器
             │ 观测: {"images": {...}, "state": ndarray}
             ▼
        arena.adapter.EmbodimentAdapter   (图像 resize / proprio 归一化)
             ▼
        arena.client.ControlLoop ──► InProcessClient (mock, 进程内随机动作)
                                  └─► PolicyClient   (HTTP POST /act)
             ▼
        UnifoLM-VLA 策略服务器 (默认 http://127.0.0.1:8777/act)
             │ ActionChunk (H 步动作块)
             ▼
        robot.execute() → env.step() → 物理步进 → 下一轮观测

  定位说明：本文件是"冒烟测试 / 联通性验证"工具，不是任务评测脚本。
  它不做成功率统计，也不构造具体操作任务，只回答一个问题：
  "ARENA 的适配器与客户端能不能把任意 Isaac Lab 环境接上 VLA 服务器"。

【依赖的外部系统】
  1. isaaclab / isaaclab_tasks (Isaac Lab 0.47.2)
     - isaaclab.app.AppLauncher        : Kit 应用生命周期
     - isaaclab_tasks.utils.parse_env_cfg : 按 env id 生成标准配置
     - import isaaclab_tasks           : 触发内置任务注册(副作用导入)
  2. gymnasium + omni.usd              : 环境注册表与 USD stage 管理
  3. arena 包: adapter / client / config / backends / types
  4. UnifoLM-VLA 策略服务器 (仅 --backend http 需要) + json_numpy
  5. NVIDIA GPU (--device cuda:0 为默认) ；用 cpu 亦可但很慢

【运行前提】
  - 已 conda activate env_isaaclab (Python 3.11)
  - 首次运行需联网(Isaac Lab 可能下载任务资产)
  - --backend http 时策略服务器需先就绪；本文件会尝试 json_numpy.patch()，
    若该包缺失只告警不退出(见 main 中 ImportError 分支)
  - 本文件不使用相机：AppLauncher 未开启 enable_cameras，因此观测里永远
    不会有真实图像，get_observation() 会走 224×224 全零占位图分支
    (与 arena_g1_microwave.py 的相机 workaround 是同一后果，但成因不同：
     这里是"没开相机"，那边是"Isaac Sim 5.1 相机 0 维 bug 被迫关闭")

【与 arena_g1_microwave.py 的关键差异】
  - 用 gymnasium 标准注册路径 gym.make(env_id, cfg=env_cfg)（那边的 Arena 配置
    无法走注册路径，只能直接实例化 ManagerBasedRLEnv）。
  - 状态观测走通用候选键搜索 STATE_KEYS（那边直取 robot_joint_pos）。
  - _unwrap() 在 obs 非 dict 时会用 {"state": obs} 包一层（那边直接用 {}），
    因此纯张量观测的环境在这里也能取回状态。
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helper: wrap an Isaac Lab env to match ARENA's expected observation format
# ------------------------------------------------------------------
# 【分节 1/6 · 环境适配器 / env-wrapper】
# 把"任意 Isaac Lab 环境"统一成 ARENA 的 RobotInterface 协议，
# 使 ControlLoop 无需关心底层是 Cartpole 还是人形操作任务。
# ------------------------------------------------------------------

class IsaacLabRobotWrapper:
    """Adapts any Isaac Lab gymnasium environment for ArenaSimRobot.

    Bridges:
    - Observation: Isaac Lab returns torch tensors → convert to numpy
    - Action: ARENA produces numpy → convert to torch for Isaac Lab
    - Key mapping: {"policy": tensor, ...} → {"images": {}, "state": ndarray}

    中文说明:
        泛化的 Isaac Lab → ARENA 环境适配器。

        做什么:
            实现 RobotInterface 所需的四个方法 reset / get_observation /
            execute / is_running，并把 Isaac Lab 的观测与动作在两种表示之间
            互相翻译。

        三条桥接路径:
          1) 观测张量: Isaac Lab 全部产出 torch(通常在 GPU) → 统一转 numpy，
             因为 ARENA 框架与 VLA 载荷(json_numpy)都按 numpy 语义工作。
          2) 动作表示: ARENA 侧是 numpy → env.step() 需要 torch 且要带 batch 维。
          3) 键名映射: Isaac Lab 的嵌套观测 {"policy": tensor, ...} → ARENA 的
             {"images": {...}, "state": ndarray}。

        设计取舍:
            - 只要求 env 满足 gymnasium 风格接口(reset/step/action_space/
              unwrapped)，因此既能接 gym.make() 包装过的环境，也能接裸的
              ManagerBasedRLEnv。
            - 与 G1MicrowaveRobot 不同，这里的动作维度在构造时直接解引用
              env.action_space.shape[-1]，不做异常兜底：本文件的定位是
              "联调工具"，维度拿不到就应该立即失败并暴露问题，而不是静默退化。
            - 图像/状态只做键名归一，不做 resize 或数值域变换 —— 那属于
              EmbodimentAdapter 的职责，避免同一件事在两处实现。

        失败语义:
            reset() 之前调用 get_observation() 抛 RuntimeError；找不到状态的
            图像时退化为占位图/空数组；env.step() 的异常向上冒泡。
    """

    STATE_KEYS = (
        "state", "robot_state", "proprio", "joint_position",
        "policy", "observation.state", "obs",
    )
    FULL_IMAGE_KEYS = (
        "full_image", "image", "agentview_image", "rgb", "head_image",
    )
    WRIST_IMAGE_KEYS = (
        "wrist_image", "robot0_eye_in_hand_image", "wrist", "hand_image",
    )
    # 为什么三张表都按"优先级顺序"逐键探测(_search 命中即返回)？
    #   本文件的卖点就是"不挑环境"，而不同 Isaac Lab 任务/不同复刻环境的观测
    #   键名完全不一致：direct 任务常用扁平键，manager-based 任务嵌在 "policy"
    #   下，robomimic 风格复刻用 agentview_image / robot0_eye_in_hand_image。
    #   写死任一个键都会让脚本换个 --env 就 KeyError，所以改为按"最具体 →
    #   最通用"排序逐个探测：具体键名(如 observation.state、full_image)语义
    #   明确，排在泛化键名(如 rgb、obs)之前，避免误取到含义不同的数组。
    #   三张表互不干扰，get_observation() 对每类各查一次。

    def __init__(self, env, instruction: str = "", max_steps: int = 1000):
        """
        初始化适配器。

        参数:
            env: 任意 Gymnasium 风格的 Isaac Lab 环境(已 gym.make 包装或裸的
                 ManagerBasedRLEnv 均可)。必须提供 action_space；观测通过
                 reset()/step() 的返回值获取。
            instruction: 自然语言任务指令，会随观测一起发给策略服务器。
            max_steps: episode 步数硬上限，is_running() 据此退出。

        返回: 无(构造函数)。

        失败会怎样:
            若 env.action_space 为 None 或形状异常，int(env.action_space.shape[-1])
            会在此处直接抛异常 —— 这是刻意不做兜底的：本文件用于联调，
            维度信息缺失应当立刻暴露，而不是带着错误的 action_dim 跑完整轮。

        实例属性:
            steps       : 已执行步数(reset 清零 / execute 自增)
            _last_obs   : 最近一次解包后的观测字典，reset 前为 None
            _action_dim : 动作向量长度，用于 execute() 的对齐与裁剪
        """
        self.env = env
        self.instruction = instruction
        self.max_steps = max_steps
        self.steps = 0
        self._last_obs: Optional[Dict[str, Any]] = None
        self._action_dim = int(env.action_space.shape[-1])

    def reset(self) -> Dict[str, Any]:
        """
        重置底层环境并返回 ARENA 规范格式的初始观测。

        返回:
            Dict[str, Any]: {"images": {...}, "state": float32 (D,)}，
            由 get_observation() 生成。

        副作用:
            覆盖 self._last_obs、把 self.steps 归零(否则上一 episode 的步数
            会继续累加，导致新一轮刚开局就因 max_steps 立刻结束)。

        实现要点:
            原始返回值先经 _unwrap() 压平再缓存，把 gym/gymnasium/Isaac Lab
            之间的返回结构差异收敛到唯一一处；同时打一条 debug 日志列出键名，
            方便换 --env 时排查"观测键到底叫什么"。
        """
        result = self.env.reset()
        self._last_obs = self._unwrap(result)
        self.steps = 0
        logger.debug("reset: _last_obs keys=%s", list(self._last_obs.keys()))
        return self.get_observation()

    def get_observation(self) -> Dict[str, Any]:
        """
        把缓存的原始观测转换为 ARENA / VLA 规范格式。

        返回:
            {
              "images": {"head": uint8 ndarray, 可选 "wrist": ndarray},
              "state":  float32 一维 ndarray(取不到时为长度 0 的空数组)
            }

        失败会怎样:
            - 未先 reset() → RuntimeError(提前失败，避免把空观测发给服务器)。
            - 找不到图像 → 不报错，注入 224×224×3 全零占位图(head)。
            - 找不到状态 → 不报错，返回长度 0 的 float32 数组，由适配器/服务端
              自行容忍；这样即便某个环境完全没有本体感知，链路也能继续联通测试。

        说明:
            键名探测顺序由 FULL_IMAGE_KEYS / WRIST_IMAGE_KEYS / STATE_KEYS
            决定，命中即停(见类属性处的注释)。
        """
        if self._last_obs is None:
            raise RuntimeError("reset() must be called before get_observation()")
        # 键名归一：把底层五花八门的键映射为 ARENA 约定的 "head"/"wrist"。
        # 只做改名与 dtype 归一，不做 resize/数值变换(那是 EmbodimentAdapter 的事)。
        images = {}
        full = self._search(self._last_obs, self.FULL_IMAGE_KEYS)
        if full is not None:
            images["head"] = full
        wrist = self._search(self._last_obs, self.WRIST_IMAGE_KEYS)
        if wrist is not None:
            images["wrist"] = wrist

        # If no camera images are available, generate a dummy placeholder.
        # The VLA server expects at least a full_image in the payload.
        # 为什么无相机时要造 224×224 全零占位图，而不是留空/报错？
        #   1) 本文件启动 AppLauncher 时未传 enable_cameras=True，默认任务
        #      (Isaac-Cartpole-Direct-v0 等) 也不含相机，所以"没有图像"是常态；
        #   2) VLA 服务端的请求协议里 full_image 是必填字段(策略骨干按固定
        #      输入通道拼接图像 token)，缺字段会直接 400/KeyError，把一次
        #      本可完成的联通性测试变成失败；
        #   3) 224×224 正是 UnifoLM-VLA 的图像输入尺寸，全零而非随机噪声，
        #      既保证结果确定性可复现，也明确表达"这里没有真实视觉信息"，
        #      让策略退化为只依赖 state 分支。
        #   代价：无相机时策略看到的永远是同一张黑图，视觉条件分支实际失效 ——
        #   这是联通性测试场景下可接受的退化。
        if not images:
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            images["head"] = dummy
            logger.debug("get_observation: no camera — using dummy 224x224 image")

        # 状态走通用候选键搜索(而不是像 G1 脚本那样直取 robot_joint_pos)：
        # 本文件要兼容任意 --env，不同任务的观测键名差异极大，只能靠候选表探测。
        # 全部落空时给长度 0 的 float32 数组而不是 None，保证下游 reshape(-1)
        # 与 to_dict() 不会因 None 崩溃。
        state = self._search(self._last_obs, self.STATE_KEYS)
        if state is None:
            state = np.zeros(0, dtype=np.float32)
        return {
            "images": images,
            "state": np.asarray(state, dtype=np.float32).reshape(-1),
        }

    def execute(self, action: np.ndarray) -> None:
        """
        执行一步动作并推进底层仿真。

        参数:
            action: 动作向量，形状 (D,) 或任何可 reshape 成 1 维的形式。

        返回:
            None。结果写入 self._last_obs 并自增 self.steps。

        处理顺序(不可交换):
            1) reshape(-1) 压平；
            2) 补齐/截断到 self._action_dim；
            3) 按 action_space.low/high 逐元素裁剪；
            4) 转 torch 张量并补 batch 维、搬到 env 所在设备；
            5) env.step()，缓存观测、步数 +1。
            先对齐维度再裁剪的原因：若先裁剪后补齐，补进去的 0 会落在裁剪之后
            从而绕过限位保护；而且长度不一致时 low/high 的广播可能静默给出错误
            结果。补 0 是因为 0 在归一化动作空间里代表"不动作/中间位"，最安全。

        失败会怎样:
            不做 try/except，env.step() 的异常向上冒泡到 main()，由 finally
            负责 env.close()/sim_app.close() 清理。
        """
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        # Truncate/pad to match env's action dim
        if len(arr) < self._action_dim:
            arr = np.pad(arr, (0, self._action_dim - len(arr)))
        elif len(arr) > self._action_dim:
            arr = arr[: self._action_dim]

        # Isaac Lab expects actions in [action_space.low, action_space.high]
        # 裁剪是最后一道安全网：VLA 输出经反归一化后可能越界，直接送入 action
        # manager 会让目标飞出可行域。min_len 取交集长度，兼容 action_space
        # 只声明了部分自由度的情况。
        if hasattr(self.env, "action_space") and self.env.action_space is not None:
            low = np.asarray(self.env.action_space.low).reshape(-1)
            high = np.asarray(self.env.action_space.high).reshape(-1)
            # Broadcast clamping
            min_len = min(len(arr), len(low))
            arr[:min_len] = np.clip(arr[:min_len], low[:min_len], high[:min_len])

        action_tensor = torch.from_numpy(arr).float().unsqueeze(0)
        # unsqueeze(0) 补 batch 维：Isaac Lab 动作形状固定为 (num_envs, D)，
        # 本脚本 num_envs=1，故为 (1, D)。设备跟随 env.unwrapped.device，
        # 避免 CPU/GPU 张量混用触发隐式拷贝或直接报错。
        if hasattr(self.env.unwrapped, "device"):
            action_tensor = action_tensor.to(self.env.unwrapped.device)

        # inference_mode: 不记录 autograd 图，省显存并避免长循环中意外累积计算图。
        with torch.inference_mode():
            result = self.env.step(action_tensor)
        self._last_obs = self._unwrap(result)
        self.steps += 1

    def is_running(self) -> bool:
        """
        判断闭环是否应继续。

        返回:
            bool: True 表示还可以继续跑。两个停止条件取"或"——
            达到 max_steps(硬上限)或底层报出终止标记(任务结束/被截断)。

        意义:
            max_steps 是保护性上限：即使策略一直输出无效动作、环境永不终止，
            也不会无限占着 GPU；同时保证脚本在无人值守的联调场景下必然退出。
        """
        running = self.steps < self.max_steps and not self.task_finished()
        return running

    def task_finished(self) -> bool:
        """
        检查底层环境是否已报告 episode 结束。

        返回:
            bool: 任一终止标记为真即返回 True；尚未 reset()(_last_obs 为 None)
                  时返回 False，即"不认为已结束"。

        检查的标记键:
            "terminated" / "truncated" / "done" / "success" —— 覆盖 gymnasium
            5 元组、旧版 gym 4 元组以及部分环境额外提供的 success 键。

        实现细节:
            用 np.asarray(value).any() 而不是 bool(value)：value 往往是形状 (1,)
            的 torch/GPU 张量，对多元素张量直接 bool() 会抛异常、对 CUDA 张量
            也会报错，先转 numpy 再 any() 最稳。任何一路为真即视为结束，
            这样"任务成功"与"超时截断"都能让闭环正确退出，不会继续往已自动
            reset 的环境里灌动作。
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

    @staticmethod
    def _to_numpy(val):
        """Safely convert a value to numpy, handling CUDA tensors.

        中文说明:
            把任意值安全地转成 numpy 数组。

            参数: val — numpy 数组 / torch 张量(CPU 或 CUDA) / 标量 / 序列。
            返回: np.ndarray；入参已是 numpy 时原样返回(不复制，避免每步多一次
                  大数组拷贝)。
            失败会怎样: 对 torch 显式 .cpu() 后再转，因为 CUDA 张量无法被
                  np.asarray 直接消费；np.asarray 首次失败时改用 float32 再试，
                  若仍失败则异常向上冒泡(这里只是把失败点后移，并非吞掉)。
        """
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, (torch.Tensor,)):
            return val.cpu().numpy()
        try:
            return np.asarray(val)
        except Exception:
            return np.asarray(val, dtype=np.float32)

    @staticmethod
    def _unwrap(result) -> Dict[str, Any]:
        """
        把 env.reset() / env.step() 的各种返回结构统一压平成字典。

        参数:
            result: 原始返回值，可能是 5 元组 / 4 元组 / 2 元组 / 字典 / 裸张量。

        返回:
            Dict[str, Any]:
              5 元组 (obs, reward, terminated, truncated, info) — gymnasium 0.29+
              4 元组 (obs, reward, done, info)                  — 经典 gym
              2 元组 (obs, info)                                — 老式
              字典   — 原样返回
              其它   — 包成 {"state": 一维 float32 数组}
            观测键与 terminated/truncated/done/reward 标记混在同一层，
            这样 get_observation() 与 task_finished() 都只需面对一种结构。

        为什么 obs 不是字典时要包成 {"state": obs}？
            有些 Isaac Lab 任务(尤其 direct 工作流)的观测就是单个张量，
            直接 dict(obs) 会抛 TypeError。包上统一键名后，_search() 里的
            STATE_KEYS 候选表(首项即 "state") 就能把它正常取出。

        失败会怎样: 不抛异常；无法识别的类型退化为 {"state": ...}，
            保证调用方总能拿到 dict。
        """
        if isinstance(result, tuple):
            if len(result) == 5:  # (obs, reward, terminated, truncated, info)
                obs, reward, terminated, truncated, info = result
                merged = dict(obs) if isinstance(obs, dict) else {"state": obs}
                merged["terminated"] = IsaacLabRobotWrapper._to_numpy(terminated)
                merged["truncated"] = IsaacLabRobotWrapper._to_numpy(truncated)
                merged["reward"] = IsaacLabRobotWrapper._to_numpy(reward)
                return merged
            if len(result) == 4:  # (obs, reward, done, info)
                obs, reward, done, info = result
                merged = dict(obs) if isinstance(obs, dict) else {"state": obs}
                merged["done"] = IsaacLabRobotWrapper._to_numpy(done)
                merged["reward"] = IsaacLabRobotWrapper._to_numpy(reward)
                return merged
            # (obs, info)
            obs, info = result[0], result[1] if len(result) > 1 else {}
            merged = dict(obs) if isinstance(obs, dict) else {"state": obs}
            if isinstance(info, dict):
                for k, v in info.items():
                    merged[k] = IsaacLabRobotWrapper._to_numpy(v) if isinstance(v, (torch.Tensor,)) else v
            return merged
        if isinstance(result, dict):
            return result
        return {"state": np.asarray(result, dtype=np.float32).reshape(-1)}

    @staticmethod
    def _search(observation: Dict[str, Any], keys) -> Optional[np.ndarray]:
        """
        按给定优先级顺序在观测字典中查找第一个可用的键。

        参数:
            observation: 扁平化的观测字典(通常来自 _unwrap())。
            keys: 候选键名序列，越靠前优先级越高。

        返回:
            Optional[np.ndarray]: 命中键对应的 numpy 数组；torch 张量会先
            .cpu().numpy()；numpy 数组原样返回；其它类型尝试 np.asarray()。
            一个都没命中时返回 None。

        实现细节:
            - 同时判断 "key in observation" 与 "值不为 None"：配置里被显式
              置 None 的观测项(例如相机被 workaround 摘除)应视为不存在，
              否则会拿到 None 继续往下走，在后面某个环节才炸。
            - np.asarray 转换失败时 continue 而不是 return None：继续尝试
              后面的候选键，避免一个"存在但格式怪异"的键掩盖掉真正可用的键。

        失败会怎样: 不抛异常，最坏返回 None，由调用方决定退化策略。
        """
        for key in keys:
            if key in observation and observation[key] is not None:
                val = observation[key]
                if isinstance(val, torch.Tensor):
                    return val.cpu().numpy()
                if isinstance(val, np.ndarray):
                    return val
                try:
                    return np.asarray(val)
                except Exception:
                    continue
        return None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
# 【分节 2/6 · 主流程 / main】依次为:
#   命令行参数 → 启动引擎 → 环境创建 → 适配器与客户端 → 闭环执行 → 清理
# 注意 main() 内的 Isaac Lab 相关 import 都写在函数体内：它们必须在
# AppLauncher 拉起 Kit 之后才可导入，放在文件顶部会在导入期就失败。
# ------------------------------------------------------------------

def main():
    """
    ARENA + Isaac Lab → VLA 策略服务器 的联通性验证入口。

    执行流程:
      ① 解析命令行参数(env id / 后端 / 设备 / 步数上限)
      ② AppLauncher 启动 Isaac Sim (Kit + PhysX；未开启相机)
      ③ 新建 USD stage 并用 gym.make(env_id, cfg=parse_env_cfg(...)) 创建环境
      ④ 构造 EmbodimentAdapter(观测编码/动作解码) 与 IsaacLabRobotWrapper
      ⑤ 依据 --backend 选择进程内 Mock 客户端或 HTTP PolicyClient
      ⑥ ControlLoop 跑 max_episodes 轮闭环，打印 JSON 结果
      ⑦ finally 中关闭环境与 Kit

    参数: 无(全部来自命令行)。
    返回: 无。结果只通过日志输出，退出码不反映策略表现。

    前置条件与边界:
      - 必须在已激活 env_isaaclab 且能访问 GPU 的机器上运行；
      - 本函数不做相机初始化，故观测中不会有真实图像(见 get_observation())；
      - 任何异常都会被宽泛捕获并记录，真正的清理在 finally 中完成。
    """
    parser = argparse.ArgumentParser(description="ARENA + Isaac Lab VLA pipeline")
    # 开关说明(全部透传到下游):
    #   --env      : 任意已注册的 Isaac Lab 任务 id，决定环境与动作维度，
    #                默认 Isaac-Cartpole-Direct-v0(最快、无需 GPU 推理即可冒烟)
    #   --backend  : mock = 进程内随机动作(不依赖服务器) / http = 对接 VLA 服务器
    #   --episodes : 闭环重复轮数，>1 可用于观察稳定性
    #   --max_steps: 单轮步数硬上限(包装器 is_running() 据此退出)
    #   --device   : cuda:0 或 cpu；Isaac Lab 在 cpu 上极慢，仅用于排障
    parser.add_argument("--env", default="Isaac-Cartpole-Direct-v0")
    parser.add_argument("--backend", default="mock", choices=["mock", "http"])
    parser.add_argument("--server_url", default="http://127.0.0.1:8777/act")
    parser.add_argument("--instruction", default="balance the pole")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no_headless", dest="headless", action="store_false")
    parser.add_argument("--device", default="cuda:0", help="Device: cuda:0 or cpu")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Launch Isaac Sim
    # ------------------------------------------------------------------
    # 【分节 3/6 · 启动引擎 / launch-engine】
    # AppLauncher 负责加载 Kit 配置、初始化 PhysX 与渲染，返回 app 句柄。
    # 这里刻意没有传 enable_cameras=True：本脚本用不到相机，且 Isaac Sim 5.1
    # 的相机管线存在 0 维数据崩溃问题，开启只会平白引入风险(参见
    # arena_g1_microwave.py 中同名 workaround)。
    # ------------------------------------------------------------------
    from isaaclab.app import AppLauncher

    logger.info("Launching Isaac Sim (headless=%s)...", args.headless)
    app_launcher = AppLauncher(headless=args.headless)
    sim_app = app_launcher.app

    # ------------------------------------------------------------------
    # 2. Create Isaac Lab environment
    # ------------------------------------------------------------------
    # 【分节 4/6 · 环境创建 / env-creation】
    # 这里走的是 gymnasium 标准注册路径(与 arena_g1_microwave.py 直接实例化
    # ManagerBasedRLEnv 不同)：本文件用的都是 Isaac Lab 自带、已注册的任务，
    # 注册路径可用且能自动完成 gym 的 wrapper 包装与观测空间校验。
    # ------------------------------------------------------------------
    import gymnasium as gym
    import omni.usd
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # 新建空 USD stage：清除上一份导入遗留的舞台内容，避免在旧舞台
    # (可能已被关闭或含残留 prim)上叠加环境导致路径冲突。
    omni.usd.get_context().new_stage()

    logger.info("Creating environment: %s", args.env)
    # use_fabric=True: 用 Fabric 承载物理状态(GPU 上的扁平化数据)，单环境也
    # 能减少 USD↔PhysX 的同步开销。
    env_cfg = parse_env_cfg(args.env, device=args.device, num_envs=1, use_fabric=True)
    # clone_in_fabric=False: 让 env 的 reset 走"写回 USD 再刷新"的常规路径。
    # 若为 True，克隆体只存在于 Fabric 中，部分自带 reset 逻辑的任务会因为
    # 读不到 USD 侧的初始位姿而复位异常(表现为 reset 后状态不归零)。
    env_cfg.scene.clone_in_fabric = False
    env = gym.make(args.env, cfg=env_cfg)
    # 禁用 Kit 的"应用停止→自动清理仿真"回调：本脚本 finally 已显式调用
    # env.close()/sim_app.close()，两条路径同时释放同一批 GPU/PhysX 资源会
    # 造成重复释放并在退出阶段崩溃。置 None 后关闭时序完全由脚本掌控。
    env.unwrapped.sim._app_control_on_stop_handle = None  # type: ignore

    action_dim = int(env.action_space.shape[-1])
    logger.info("Obs space: %s", env.observation_space)
    logger.info("Action space: %s (dim=%d)", env.action_space, action_dim)

    # ------------------------------------------------------------------
    # 3. ARENA adapter + client
    # ------------------------------------------------------------------
    # 【分节 5/6 · 适配器与客户端 / adapter-and-client】
    # ------------------------------------------------------------------
    from arena.adapter import EmbodimentAdapter
    from arena.client import ControlLoop, PolicyClient
    from arena.config import AdapterConfig, ClientConfig

    # Match adapter config to the actual environment
    # robot_type="isaaclab_generic"：告诉适配器不要套用某个具体机型(如 unitree_g1)
    # 的归一化常量，而是采用通用约定 —— 因为本文件可能接 Cartpole、Go2 等
    # 完全不同的动作空间。action_dim 必须与 env 实际维度一致，否则适配器
    # 解码出的动作块会被包装器补齐/截断，测出来的就不是真实链路行为。
    adapter = EmbodimentAdapter(
        AdapterConfig(
            robot_type="isaaclab_generic",
            action_dim=action_dim,
        ),
        instruction=args.instruction,
    )

    robot = IsaacLabRobotWrapper(
        env, instruction=args.instruction, max_steps=args.max_steps
    )

    # Ensure json_numpy is patched for HTTP client (VLA server uses it)
    # 为什么 HTTP 分支必须先 patch？
    #   ARENA 的观测/动作载荷里含 numpy 数组，标准 json 无法序列化。VLA 服务端
    #   约定用 json_numpy 的扩展编码，客户端不 patch 就会在 requests 编码阶段
    #   抛 TypeError。这里对 ImportError 只告警不退出，是为了让"没装 json_numpy"
    #   的场景仍能跑起来并在真正的请求处暴露问题(而非在启动阶段就 fail-fast)。
    if args.backend == "http":
        try:
            import json_numpy

            json_numpy.patch()
            logger.debug("json_numpy patch applied for HTTP transport")
        except ImportError:
            logger.warning("json_numpy not available; HTTP transport may fail with numpy payloads")

    if args.backend == "mock":
        from arena.backends import build_backend
        from arena.config import ServerConfig
        from arena.types import ActionChunk

        # Use mock backend matched to action dim
        # 为什么必须手动覆盖 action_dim？
        #   ServerConfig(backend="mock") 只选定后端实现，维度仍用类内默认值 7
        #   (单臂 Franka 风格)。不覆盖的话 mock 只会吐 7 维随机动作，经包装器
        #   对齐后变成"前 7 维随机 + 其余恒 0"，机器人大部分自由度不动，
        #   会被误判成环境故障。这里对齐到 env 的真实维度，才能让 mock 覆盖
        #   全部自由度。注意该属性在首次 infer 时才被读取，故必须在建立
        #   client 之前赋值。
        backend = build_backend(ServerConfig(backend="mock"))
        backend.action_dim = action_dim  # override default (7)

        # 进程内客户端：与 PolicyClient 保持同一个 duck-typing 接口
        # (infer(observation, instruction) -> ActionChunk)，从而让 ControlLoop
        # 完全不必区分"策略在进程内还是在远端"。observation.to_dict() 是 ARENA
        # Observation 对象转载荷字典的方法。
        class InProcessClient:
            """
            进程内策略客户端 (mock 专用)。

            作用: 把 mock 后端的 backend.infer(list[dict], instruction) 接口
            包装成 ControlLoop 期望的 infer(observation, instruction) → ActionChunk
            接口。这样闭环代码不需要任何 mock/HTTP 分支判断，两者可互换。
            """

            def infer(self, observation, instruction=None):
                """
                执行一次 mock 推理。

                参数:
                    observation: ARENA Observation 对象，需实现 to_dict()。
                    instruction: 指令文本；None 时退化为空串。

                返回:
                    ActionChunk: 内含 backend 生成的 (H, action_dim) 动作块。

                说明: mock 后端按批量接口设计，故把单条观测包成 [ ... ] 传入。
                """
                actions = backend.infer([observation.to_dict()], instruction or "")
                return ActionChunk(actions=actions)

        client = InProcessClient()
    else:
        # HTTP 后端：把观测 POST 到 VLA 服务器的 /act，取回动作块。
        client = PolicyClient(ClientConfig(server_url=args.server_url))

    # ------------------------------------------------------------------
    # 4. Run
    # ------------------------------------------------------------------
    # 【分节 6/6 · 闭环执行与清理 / run-and-teardown】
    # ControlLoop 的标准闭环: 观测 → 适配器编码 → 策略推理 → 解码动作块
    # → 逐动作 execute()，直到 robot.is_running() 为假。max_episodes 来自
    # 命令行，允许重复多轮以观察稳定性。
    # ------------------------------------------------------------------
    logger.info("Starting control loop (%d episode(s))...", args.episodes)
    loop = ControlLoop(
        robot, client, adapter,
        instruction=args.instruction,
        max_episodes=args.episodes,
    )

    try:
        logger.info(">>> Running episode...")
        # 每处 sys.stdout.flush() 都是为了在 Isaac Sim 频繁输出或崩溃时仍能及时
        # 看到 Python 侧的进度日志：Kit 会接管 stdout 缓冲，不主动 flush 时日志
        # 可能被吞掉，尤其在进程异常退出(段错误)的场景下会丢掉全部线索。
        sys.stdout.flush()
        results = loop.run(args.instruction)
        logger.info(">>> Episode complete!")
        logger.info("Results: %s", json.dumps(results, indent=2))
        sys.stdout.flush()
    except Exception as e:
        # 宽泛捕获是刻意的：即使闭环失败也必须走到 finally 关闭 Kit，
        # 否则会留下占用显存的僵尸进程，让后续运行因 VRAM 不足而失败。
        # logger.exception 会打印完整堆栈，便于区分是环境、适配器还是服务端的问题。
        logger.exception("Error during control loop: %s", e)
        sys.stdout.flush()
    finally:
        # 关闭顺序不可颠倒：先关环境(释放 PhysX/渲染资源)，再关 Kit 应用本身。
        # 反过来会让 env 在已销毁的 Kit 上下文里做清理，触发绑定层崩溃。
        logger.info(">>> Closing environment...")
        sys.stdout.flush()
        env.close()
        logger.info(">>> Closing Isaac Sim...")
        sys.stdout.flush()
        sim_app.close()
        logger.info("Shutdown complete.")
        sys.stdout.flush()


# ------------------------------------------------------------------
# 【入口 / entry-point】
# 只有直接执行本文件时才启动仿真。被 import 时(例如在单元测试里复用
# IsaacLabRobotWrapper)不产生任何副作用 —— 这点尤其重要：main() 会启动
# Kit，若在 import 期启动会污染同一进程内的其它 Isaac Lab 测试。
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()