# ARENA 0.1.0 — 完整技术文档

> **版本**: 0.1.0  
> **更新日期**: 2026-09-17  
> **适用范围**: `arena_0_1_0` 项目全部模块  

---

## 目录

1. [项目层次结构](#一项目层次结构)
2. [模块调用关系](#二模块调用关系)
3. [ARENA 框架详解](#三arena-框架详解)
4. [Isaac Lab Arena 0.1.0 启动过程](#四isaac-lab-arena-010-启动过程)
5. [核心三模块编程实现](#五核心三模块编程实现)
   - 5.1 [Scene — 场景编排](#51-scene--场景编排)
   - 5.2 [Embodiment — 智能体机器人](#52-embodiment--智能体机器人)
   - 5.3 [Task — 任务逻辑](#53-task--任务逻辑)
6. [官方示例代码详细运行过程](#六官方示例代码详细运行过程)
7. [归一化系统](#七归一化系统)
8. [Sim-to-Real 域随机化](#八sim-to-real-域随机化)
9. [CLI 命令速查](#九cli-命令速查)
10. [集成脚本：G1-D 厨房微波炉](#十集成脚本g1-d-厨房微波炉)

---

## 一、项目层次结构

```
arena_0_1_0/                          # 项目根目录
│
├── arena/                            # ── ARENA 框架 (0.1.0) ──
│   ├── __init__.py                   #    包入口，导出核心类型
│   ├── types.py                      #    Observation / ActionChunk 规范
│   ├── config.py                     #    配置类 + 归一化/反归一化工具
│   ├── adapter.py                    #    EmbodimentAdapter (观测/动作适配器)
│   ├── backends.py                   #    可插拔策略后端 (mock/unifolm_vla/openpi/http)
│   ├── server.py                     #    FastAPI 策略服务器 (/act, /health)
│   ├── client.py                     #    HTTP 客户端 + 仿真/真机控制回路
│   ├── sim2real.py                   #    Sim-to-Real 域随机化课程 (L1–L4)
│   ├── cli.py                        #    命令行入口 (arena server|sim|real)
│   └── configs/
│       └── default.yaml              #    默认配置文件
│
├── IsaacLab/                         # ── Isaac Lab 仿真基础设施 ──
│   ├── source/
│   │   ├── isaaclab/                 #    Isaac Lab 核心 (AppLauncher, envs, managers)
│   │   ├── isaaclab_assets/          #    资产 (机器人 URDF/USD 引用)
│   │   ├── isaaclab_tasks/           #    任务 (Gym 注册, ManagerBased)
│   │   └── isaaclab_rl/              #    强化学习库
│   ├── isaaclab-arena-envs/          #    Arena EnvHub 桥接层 (LeRobot 兼容)
│   │   ├── env.py                    #      make_env (EnvHub API)
│   │   ├── isaaclab_env_wrapper.py   #      IsaacLabEnvWrapper
│   │   └── example_envs.yaml         #      环境注册表
│   ├── apps/                         #    KIT 应用配置文件
│   ├── scripts/                      #    示例脚本 (demos, RL, 遥操作)
│   └── tools/                        #    安装/测试工具
│
├── ../unitree_G1D/Arena_0_1_0/       # ── Isaac Lab Arena SDK ──
│   └── IsaacLab-Arena/
│       └── isaaclab_arena/
│           ├── assets/               #    资产系统
│           │   ├── asset.py          #      Asset 基类
│           │   ├── asset_registry.py #      AssetRegistry / DeviceRegistry
│           │   ├── background_library.py   # 背景库 (kitchen, warehouse...)
│           │   └── object_library.py       # 物体库 (microwave, cabinet...)
│           ├── scene/
│           │   └── scene.py          #      Scene 组合器
│           ├── embodiments/          #    智能体
│           │   ├── embodiment_base.py       # EmbodimentBase 基类
│           │   ├── gr1t2/            #      GR1T2 (Fourier GR-1 双臂人形)
│           │   ├── g1/               #      G1 (宇树) 动作配置
│           │   └── franka/           #      Franka Panda 机械臂
│           ├── tasks/                #    任务系统
│           │   ├── task_base.py             # TaskBase 基类
│           │   └── open_door_task.py         # OpenDoorTask (开门任务)
│           ├── environments/         #    环境构建
│           │   ├── arena_env_builder.py      # ArenaEnvBuilder
│           │   ├── isaaclab_arena_environment.py  # IsaacLabArenaEnvironment
│           │   └── isaaclab_arena_manager_based_env.py  # 管理型环境配置
│           ├── examples/             #    官方示例
│           │   └── example_environments/
│           │       ├── example_environment_base.py  # 示例基类
│           │       ├── gr1_open_microwave_environment.py  # ★ G1 微波炉
│           │       └── ...
│           ├── affordances/          #    可交互属性 (Openable, Grabbable)
│           ├── orchestrator/         #    编排器
│           ├── policy/               #    策略接口
│           ├── teleop_devices/       #    遥操作设备
│           ├── metrics/              #    评估指标
│           ├── cli/                  #    命令行工具
│           └── utils/                #    工具函数
│
├── unifolm-vla/                      # ── VLA 模型与推理 ──
│   ├── src/unifolm_vla/              #    模型源码
│   ├── models/                       #    模型 checkpoints
│   │   ├── UnifoLM-VLA-Base1/        #    基础 VLA (action_dim=23)
│   │   ├── UnifoLM-VLA-Libero/       #    LIBERO 微调版
│   │   └── UnifoLM-VLM-Base/         #    纯 VLM (视觉语言模型)
│   ├── deployment/
│   │   └── model_server/
│   │       └── run_real_eval_server.py  #  官方 VLA 服务器 (参考实现)
│   └── experiments/                  #    实验脚本
│
├── tests/                            # ── 测试 ──
│   └── test_arena.py
│
├── arena_g1_microwave.py             # ★ G1-D 厨房微波炉终端到终端脚本
├── arena_isaaclab_demo.py            #    Isaac Lab 环境通用接入
├── demo.py                           #    ARENA 框架端到端演示
├── pyproject.toml                    #    打包配置 (arena 0.1.0)
├── README.md                         #    快速上手 (英文)
├── SUMMARY.md                        #    工作成果总结
└── DOCUMENTATION.md                  #    本文件
```

---

## 二、模块调用关系

### 2.1 仿真闭环数据流 (G1-D 微波炉)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          main() [arena_g1_microwave.py]             │
│                                                                     │
│  ① AppLauncher  →  启动 Isaac Sim (Kit + PhysX + Rendering)        │
│       │                                                             │
│  ② AssetRegistry.get_asset_by_name("kitchen")()   ─┐               │
│     AssetRegistry.get_asset_by_name("microwave")() ─┤ Scene.compose │
│     AssetRegistry.get_asset_by_name("gr1_pink")()  ─┘              │
│       │                                                             │
│  ③ OpenDoorTask(microwave, openness_threshold=0.8)                  │
│       │                                                             │
│  ④ IsaacLabArenaEnvironment(name, embodiment, scene, task)          │
│       │                                                             │
│  ⑤ ArenaEnvBuilder(arena_env, args)                                 │
│       ├─ orchestrator.orchestrate()  ← 资产间协调                   │
│       ├─ compose_manager_cfg()       ← 生成 Isaac Lab 配置类        │
│       └─ fix: remove last_action 观测项                             │
│       │                                                             │
│  ⑥ ManagerBasedRLEnv(cfg=cfg)    ← 直接实例化 (绕过 gym 注册)       │
│       │                                                             │
│  ⑦ G1MicrowaveRobot(env)            ← ARENA RobotInterface 适配器   │
│       │                                                             │
│  ⑧ ControlLoop(robot, client, adapter).run()                        │
│       └─ while is_running:                                         │
│            obs  ← robot.get_observation()                           │
│            vla_obs ← adapter.encode_observation(obs)                │
│            chunk  ← client.infer(vla_obs)   ← HTTP / Mock          │
│            decoded ← adapter.decode_chunk(chunk)                    │
│            for action in decoded: robot.execute(action)             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 服务器端数据流

```
┌─────────────────────────────────────────────────────┐
│  PolicyServer.serve(host, port)                     │
│                                                     │
│  POST /act                                          │
│    payload = {                                      │
│      "observations": [Obs, ...],                    │
│      "instruction": "open the microwave door"       │
│    }                                                │
│        │                                            │
│        ▼                                            │
│  PolicyBackend.infer([Observation, ...], instruction)│
│    ├─ UnifoLMVLABackend                             │
│    │    ├─ Qwen chat template                       │
│    │    ├─ normalize proprio                        │
│    │    ├─ DiT action head                          │
│    │    └─ unnormalize actions                      │
│    └─ MockPolicyBackend (确定性动作)                 │
│        │                                            │
│        ▼                                            │
│  → {"actions": [[...], ...], "latency_s": 0.05}     │
└─────────────────────────────────────────────────────┘
```

### 2.3 模块依赖图

```
arena.types (Observation, ActionChunk)
    ↑
arena.config (AdapterConfig, normalize, unnormalize)
    ↑
arena.adapter (EmbodimentAdapter, ObservationAdapter, ActionAdapter)
    ↑
arena.backends (PolicyBackend → Mock / UnifoLMVLA / OpenPI / HTTP)
    ↑
arena.server (PolicyServer)     arena.client (PolicyClient, ControlLoop)
                                      ↑
                              arena.cli (命令行入口)
                                      ↑
                              arena.sim2real (Curricum + Randomizer)

  ─ ─ ─ ─ ─ 横向桥接 ─ ─ ─ ─ ─

arena_g1_microwave.py
    ├─ arena.adapter.EmbodimentAdapter
    ├─ arena.client.ControlLoop, PolicyClient
    └─ isaaclab_arena.*  (Scene, Embodiment, Task, ArenaEnvBuilder)
         └─ isaaclab.envs.ManagerBasedRLEnv
```

---

## 三、ARENA 框架详解

### 3.1 arena/types.py — 规范数据结构

| 类 | 关键字段 | 用途 |
|---|---|---|
| `Observation` | `images: Dict[str, ndarray]` | 多相机帧 (H, W, 3) uint8 |
| | `state: ndarray` | 本体感知向量 |
| | `instruction: str` | 自然语言任务描述 |
| | `task_name: Optional[str]` | 归一化统计关键字段 |
| `ActionChunk` | `actions: ndarray (H, D)` | 动作块，1D 输入自动升为 2D |

**序列化 API**: `Observation.to_dict()` / `Observation.from_dict(payload)`

**图像查找属性**:
- `full_image` → keys `("full", "head", "image", "agentview_image")`
- `wrist_images` → keys containing `"wrist"`
- `all_images` → `[full_image, *wrist_images]`

### 3.2 arena/config.py — 配置与归一化

**核心类**:

| 配置类 | 关键字段 |
|---|---|
| `AdapterConfig` | `action_dim=7`, `proprio_dim=8`, `image_size=224`, `normalization_type=bounds_q99` |
| `ServerConfig` | `backend="unifolm_vla"`, `ckpt_path`, `port=8777` |
| `ClientConfig` | `server_url`, `retries=3`, `timeout_s=10` |
| `Sim2RealConfig` | 域随机化参数 (光照、纹理、相机、动力学) |

> ⚠️ **`AdapterConfig.action_dim` / `proprio_dim` 不参与任何计算**。适配器的裁剪/填充
> 一律以传入数组自身的形状为准（`apply_joint_limits()` 取 `min(action, lower, upper)`）。
> 默认值 `7 / 8` 来自 LIBERO checkpoint，仅为占位。**各实施例的真实维度见下表。**

#### 各实施例维度对照（唯一权威口径）

| 实施例 | 动作维度 | 本体感知维度 | 组成 | 权威来源 |
|---|---|---|---|---|
| GR1T2 厨房微波炉 | **36** | 变长 | 14 双臂 IK + 22 手指关节 | `arena_g1_microwave.py` 运行时 `env.action_space` |
| G1 移动操作 PnP | **23** | 43 | 2 导航 + 3+4 左 EEF + 3+4 右 EEF + 3 导航子目标 + 1 底座高度 + 3 躯干 RPY | `DOCUMENTATION.md §12.2`、`arena_g1_locomanip_pnp*.py` |
| G1 WBC 全身（仿真侧声明） | **~50+** | 变长 | 下肢 12 + 双臂 IK 14 + 手部 + 导航 | `env.action_space`（动态确定） |
| UnifoLM-VLA checkpoint 训练常量 | `G1_EE_6D`=**23** / `G1`=16 / `LIBERO`=7 / `ALOHA`=14 / `BRIDGE`=7 | 同左 | — | `rlds_dataloader/constants.py` |

> **为什么同一份文档里会出现 36 和 23？** 二者是**不同实施例**，不是同一个数字的两种写法：
> GR1T2（微波炉任务）是 36 维；G1 移动操作（箱子抓放）是 23 维。
>
> **训练侧的权威口径是 `constants.py`，不是 YAML。** `FlowmatchingActionHead`
> 直接取 `ACTION_DIM` / `PROPRIO_DIM` / `NUM_ACTIONS_CHUNK` 三个全局常量
> （`DiT_ActionHeader.py`），`constants.py` 依据命令行参数自动探测平台、
> 无匹配时回退 `G1_EE_6D`（23 / 23 / chunk=25）。
> `unifolm_vla_train.yaml` 中的 `action_dim` / `state_dim` / `action_horizon` /
> `future_action_window_size` **四个键不被任何代码读取**，已在该 YAML 内标注说明。

**归一化函数**:

```python
normalize(x, stats, norm_type)   # x → bound/bound_q99/normal
unnormalize(x, stats, norm_type) # 反向
```

| 归一化类型 | 公式 |
|---|---|
| `bounds` | `x_norm = 2*(x - low)/(high - low) - 1` |
| `bounds_q99` | 同上，但 low/high 使用 q01/q99 分位数 |
| `normal` | `x_norm = (x - mean)/(std + 1e-8)` |

mask=False 的维度原样透传，归一化结果裁剪到 [-1, 1]。

### 3.3 arena/adapter.py — 具身适配器

**三层适配器结构**:

```
EmbodimentAdapter (顶层)
  ├─ ObservationAdapter
  │    ├─ resize_image() → 纯 NumPy 双线性插值 224×224
  │    ├─ encode_state() → normalize(proprio, stats)
  │    └─ encode_observation() → Observation
  └─ ActionAdapter
       ├─ decode_action() → unnormalize + joint_limit clamp
       └─ decode_chunk() → ActionChunk
```

**关键方法**:

```python
adapter.encode(images, state, instruction=None, task_name=None) → Observation
adapter.decode(action) → np.ndarray  # 反归一化 + 关节限位
adapter.encode_observation({"images": {...}, "state": arr})  → Observation
adapter.decode_chunk(chunk_actions) → ActionChunk
```

### 3.4 arena/backends.py — 可插拔后端

| 后端 | 类 | 说明 |
|---|---|---|
| `mock` | `MockPolicyBackend` | 确定性推理，用于测试/演示 (无需 GPU) |
| `unifolm_vla` | `UnifoLMVLABackend` | 本地 UnifoLM-VLA checkpoint (Qwen + DiT) |
| `openpi` | `OpenPIBackend` | HTTP 转发到 OpenPI 服务 |
| `http` | `HTTPForwardBackend` | 透明的 HTTP 转发代理 |

**后端接口**:

```python
class PolicyBackend(abc.ABC):
    def infer(self, observations, instruction) → ActionChunk
    def health(self) → bool        # 真实探活, 不得抛异常
    def status(self) → dict        # {"backend", "healthy", "detail", ...}
```

**各后端的 `health()` 语义**（不是恒 `True`）:

| 后端 | 探活方式 | 不健康的情形 |
|---|---|---|
| `mock` | 恒健康 | —（无外部依赖） |
| `unifolm_vla` | 检查 checkpoint 是否已加载到设备 | 模型未加载 |
| `openpi` / `http` | 真实 HTTP 探测上游 | 上游不可达，**或上游 `/health` 返回 5xx** |

> **探测策略**（`probe_http_endpoint`）：先 `GET <base>/health`。若上游实现了该路由，
> 它的结论就是权威结论 —— **包括否定结论**：返回 5xx 说明上游自述不可用，转发后端
> 不能把它美化成"健康"。若上游没有该路由（如官方 `run_real_eval_server.py` 早期版本），
> 退化为 `HEAD <url>`，此时任何 2xx/4xx 都算存活（因为 `/act` 是 POST-only，回 405 是正常的）。
> **兜底探测拒绝 5xx**：代理/网关对无人监听的端口常回 502，若接受就会掩盖真正需要暴露的故障。

**注册机制**: `build_backend(ServerConfig)` 通过后端工厂函数自动路由。

### 3.5 arena/server.py — FastAPI 策略服务器

**端点**:

| 路由 | 方法 | 返回 | 功能 |
|---|---|---|---|
| `/act` | POST | `200` | 接收 `{"observations": [...], "instruction": "..."}`, 返回 `{"actions": [[...]], "latency_s": 0.05}` |
| `/health` | GET | `200` 健康 / `503` 不健康 | **真实探活**: 会调用 `backend.status()`, 而非恒返回 ok |

> `/health` 返回体示例（健康）：
> ```json
> {"status":"ok","healthy":true,"version":"0.1.0","backend":"MockPolicyBackend","detail":null}
> ```
> 不健康时 `status` 变为 `"unavailable"`，`healthy` 为 `false`，并附带 `detail`（如
> `"upstream unhealthy (GET http://.../health -> 503)"`）。编排器可直接以 HTTP 状态码
> 决定是否放流量。

> **`act()` 返回 JSON 原生类型**：动作已 `tolist()`，不再返回 `numpy.ndarray`。
> 旧实现之所以能工作，是因为 `json_numpy.patch()` 会全局替换 `json.dumps`——
> 而该 patch 只是解码请求的副作用，任何未发送 numpy 载荷的调用方都会踩到
> `TypeError: Object of type ndarray is not JSON serializable`。现在由服务器
> 自己保证（见 `_ensure_json_numpy()`）。

**启动**:

```bash
arena server --backend unifolm_vla --ckpt_path <path> --port 8777
# 等效于
python -c "from arena.server import serve; serve(ServerConfig(backend='unifolm_vla', ...))"
```

### 3.6 arena/client.py — 客户端与控制回路

**PolicyClient** (`arena/client.py`):

```python
client = PolicyClient(ClientConfig(server_url="http://127.0.0.1:8777/act"))
chunk = client.infer(observation, instruction="pick up the cube")
# → ActionChunk(actions=ndarray(H, D), metadata={'latency_s': 0.05})
```

**RobotInterface 抽象接口**:

```python
class RobotInterface(abc.ABC):
    def reset(self) -> Dict[str, Any]       # 返回 {'images': {...}, 'state': ndarray}
    def get_observation(self) -> Dict       # 获取当前观测
    def execute(self, action: ndarray)      # 执行单个动作
    def is_running(self) -> bool            # 是否继续运行
    def task_finished(self) -> bool         # 任务是否完成
```

**两个实现**:

| 实现 | 说明 |
|---|---|
| `ArenaSimRobot` | 包装 Isaac Lab / Arena gymnasium 环境，兼容 3 种 step 返回格式 |
| `UnitreeRobot` | 包装 Unitree SDK (注入式，无需具体 SDK 版本) |

**ControlLoop** (统一闭环):

```python
loop = ControlLoop(robot, client, adapter, instruction="open the microwave door")
results = loop.run()  # → [{"success": True}, ...]
```

### 3.7 arena/sim2real.py — 域随机化

**四级课程**（数值见 §8.2，此处只标是否启用）:

| Level | 光照 | 纹理 | 相机 | 动力学 | 观测噪声 | 动作延迟 |
|---|---|---|---|---|---|---|
| L1 | × | × | × | × | × | × |
| L2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| L3 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| L4 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

> **注意**：L2–L4 的六个因子**全部启用**，只是幅度不同；早期文档把 L2 的相机/动力学
> 标为「×」、L3/L4 标为「light/full」，与代码不符，已按 `CURRICULUM` 更正。
> 其中**动力学只输出参数**（需仿真器施加），见 §8.3。

**API**:

```python
from arena.sim2real import Sim2RealCurriculum, Randomizer

curriculum = Sim2RealCurriculum(Sim2RealConfig(level=2, seed=42))
curriculum.advance()                                # 下一级
curriculum.reset(level=1)                           # 重置到指定级别
curriculum.randomize_observation(obs)               # L + T + C + mu
curriculum.randomize_dynamics()                     # M: 返回缩放系数
curriculum.delay_action(action)                     # N
```

---

## 四、Isaac Lab Arena 0.1.0 启动过程

### 完整启动时序 (6 步)

```
Step 1: AppLauncher
  ├─ 加载 Kit 应用配置 (.kit 文件)
  ├─ 初始化 PhysX 物理引擎
  ├─ 启动 Hydra 渲染器
  └─ 返回 sim_app 句柄

Step 2: Scene 构建
  ├─ AssetRegistry()  ← 单例，自动注册所有已发现的资产
  ├─ get_asset_by_name("kitchen")()    → KitchenBackground
  ├─ get_asset_by_name("microwave")()  → Microwave (Openable)
  ├─ get_asset_by_name("gr1_pink")()   → GR1T2PinkEmbodiment (enable_cameras=True)
  └─ Scene(assets=[background, microwave])

Step 3: Task 构建
  ├─ OpenDoorTask(microwave,
  │       openness_threshold=0.8,   ← 80% 打开度 → 成功
  │       reset_openness=0.2,       ← 重置时关到 20%
  │       episode_length_s=5.0)
  └─ 内部生成了:
       ├─ termination_cfg (success: microwave.is_open)
       ├─ events_cfg (reset: microwave.close, set_object_pose)
       └─ metrics (SuccessRate, DoorMovedRate)

Step 4: IsaacLabArenaEnvironment
  ├─ 组合 scene + embodiment + task
  └─ 作为 arena 环境的输入

Step 5: ArenaEnvBuilder
  ├─ orchestrator.orchestrate()
  │    └─ 资产间协调 (场景碰撞组、相机注册到 embodiment)
  ├─ compose_manager_cfg()
  │    ├─ observation_cfg  ← Policy + Camera (robot_pov_cam)
  │    ├─ actions_cfg      ← PinkInverseKinematicsActionCfg (PINK IK)
  │    ├─ events_cfg       ← 重置事件 (复位关节、关闭门、重设位置)
  │    ├─ termination_cfg  ← 超时 + 成功
  │    ├─ scene_cfg        ← 物体 + 机器人 + 相机
  │    ├─ rewards_cfg      ← 空 (手工任务)
  │    └─ curriculum_cfg   ← 空 (手工任务)
  └─ 导出 IsaacLabArenaManagerBasedRLEnvCfg

Step 6: ManagerBasedRLEnv
  ├─ ManagerBasedRLEnv(cfg=cfg)  ← 直接实例化
  ├─ 初始化:
  │    ├─ Scene → 克隆到 Fabric (GPU)
  │    ├─ Observations Manager
  │    ├─ Actions Manager (PINK IK)
  │    ├─ Events Manager
  │    └─ Terminations Manager
  └─ 返回 env (gymnasium 兼容)
```

### 启动过程涉及的配置系统

```
arena_g1_microwave.py
  → argparse.Namespace (headless, enable_cameras, device, num_envs, ...)
  → AssetRegistry (自动发现)
  → IsaacLabArenaEnvironment
  → ArenaEnvBuilder
       → IsaacLabArenaManagerBasedRLEnvCfg  ← configclass (dataclass)
            ├─ ViewerCfg           ← 相机视角
            ├─ SimulationCfg        ← 物理配置 (dt=0.005, gravity=-9.81)
            ├─ SceneCfg             ← 场景 (物体 + 机器人 + 相机)
            ├─ ObservationCfg       ← 观测 (Policy + Camera)
            ├─ ActionsCfg           ← 动作 (PINK IK)
            ├─ EventsCfg            ← 事件 (重置)
            ├─ TerminationCfg       ← 终止 (超时 + 成功)
            └─ Metrics              ← 指标 (SuccessRate, DoorMovedRate)
       → ManagerBasedRLEnv(cfg=cfg)
```

---

## 五、核心三模块编程实现

### 5.1 Scene — 场景编排

**文件**: `isaaclab_arena/scene/scene.py`

**类结构**:

```python
class Scene:
    def __init__(self, assets: list[Asset] | None = None)
        self.assets: dict[str, Asset] = {}     # 资产名称 → 资产对象

    def add_asset(self, asset: Asset)           # 添加单个资产
    def add_assets(self, assets: list[Asset])   # 批量添加
    def get_scene_cfg(self) → configclass       # 生成场景配置
                                                # 通过 make_configclass 动态组合

    # 可覆盖的配置钩子:
    observation_cfg: Any = None
    events_cfg: Any = None
    termination_cfg: Any = None
    rewards_cfg: Any = None
    curriculum_cfg: Any = None
    commands_cfg: Any = None
```

**使用示例**:

```python
from isaaclab_arena.scene.scene import Scene

# 背景
kitchen = asset_registry.get_asset_by_name("kitchen")()

# 物体
microwave = asset_registry.get_asset_by_name("microwave")()
microwave.set_initial_pose(Pose(
    position_xyz=(0.4, -0.00586, 0.22773),
    rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
))

# 组合
scene = Scene(assets=[kitchen, microwave])

# 自动生成配置:
# SceneCfg(
#   kitchen=AssetBaseCfg(prim_path="/World/envs/.../kitchen", ...),
#   microwave=ArticulationCfg(prim_path="/World/envs/.../microwave", ...)
# )
```

### 5.2 Embodiment — 智能体机器人

**层级结构**:

```
EmbodimentBase (基类)
  ├─ set_initial_pose(pose)      ← 设定初始位姿
  ├─ get_cfgs() → dict           ← 返回机器人配置 (Articulation + Camera + ...)
  ├─ get_actions_cfg() → config  ← 返回动作配置
  ├─ get_observation_cfg() → config  ← 返回观测配置 (含 GR1T2ObservationsCfg)
  └─ get_events_cfg() → config   ← 返回事件配置 (复位关节)

GR1T2PinkEmbodiment ← GR-1 双臂人形 + PINK IK 控制器
  ├─ 14 自由度双臂: shoulder_pitch/roll/yaw + elbow_pitch + wrist_yaw/roll/pitch × 2
  ├─ 22 自由度手部 (Fourier 手)
  ├─ PINK IK 控制器:
  │    ├─ FrameTask(left_hand_pitch_link)  ← 左 EEF 目标
  │    ├─ FrameTask(right_hand_pitch_link) ← 右 EEF 目标
  │    ├─ DampingTask                     ← 阻尼
  │    └─ NullSpacePostureTask            ← 零空间姿态最优
  └─ 重力补偿 (enable_gravity_compensation=True)
```

**使用示例**:

```python
embodiment = asset_registry.get_asset_by_name("gr1_pink")(
    enable_cameras=True  # 启用头戴相机 → robot_pov_cam
)
embodiment.set_initial_pose(Pose(
    position_xyz=(-0.4, 0.0, 0.0),    # 机器人在微波炉左侧
    rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
))
```

**生成的关键配置**:

```
Robot ArticulationCfg:
  usd_path = "https://.../GR1T2_fourier_hand_6dof.usd"
  actuators:
    head:    ImplicitActuator (head_.*)
    trunk:   ImplicitActuator (waist_.*)
    legs:    ImplicitActuator (.*hip_.*|.*knee_.*|.*ankle_.*)
    right-arm: ImplicitActuator (effort_limit=inf, velocity_limit=inf)  ← IK 控制
    left-arm:  ImplicitActuator (effort_limit=inf, velocity_limit=inf)  ← IK 控制
    right-hand: ImplicitActuator (R_.*)
    left-hand:  ImplicitActuator (L_.*)

Camera TiledCameraCfg:
  prim_path = "/Robot/head_yaw_link/RobotPOVCam"
  resolution = (512, 512)
  data_types = ["rgb"]

Actions PinkInverseKinematicsActionCfg:
  14 个 IK 控制关节 + 22 个手部关节
  urdf_path = "/tmp/urdf/GR1T2_fourier_hand_6dof.urdf"
```

**观测输出 (GR1T2ObservationsCfg)**:

| 观测项 | 来源 | 维度 |
|---|---|---|
| `robot_joint_pos` | 所有关节位置 | 变长 |
| `robot_root_pos` | 机器人根位置 | 3 |
| `robot_root_rot` | 机器人根四元数 | 4 |
| `robot_links_state` | 关键连杆状态 | 变长 |
| `left_eef_pos / quat` | 左末端执行器位姿 | 3 + 4 |
| `right_eef_pos / quat` | 右末端执行器位姿 | 3 + 4 |
| `hand_joint_state` | 手部关节 (R_.*, L_.*) | 变长 |
| `head_joint_state` | 头部关节 (3 个) | 3 |
| `robot_pov_cam_rgb` | 头戴相机 RGB | (512, 512, 3) |

### 5.3 Task — 任务逻辑

**文件**: `isaaclab_arena/tasks/open_door_task.py`

**类结构**:

```python
class TaskBase(ABC):
    def get_termination_cfg() → config     # 终止条件
    def get_events_cfg() → config          # 事件 (重置)
    def get_metrics() → list[MetricBase]   # 评估指标
    def get_viewer_cfg() → ViewerCfg       # 查看器配置
    def get_mimic_env_cfg(embodiment_name) → MimicEnvCfg  # 模仿学习配置

class OpenDoorTask(TaskBase):
    def __init__(
        self,
        openable_object: Openable,              # ★ 可开门物体 (如 Microwave)
        openness_threshold: float | None,        # 打开度阈值 → 成功
        reset_openness: float | None,            # 重置时的打开度
        episode_length_s: float | None,          # episode 时长
    )
```

**OpenDoorTask 内部实现**:

```python
# 构建时自动生成:
self.termination_cfg = TerminationsCfg(
    time_out = TerminationTermCfg(func=isaaclab.time_out),
    success  = TerminationTermCfg(func=microwave.is_open, params={"threshold": 0.8})
)

self.events_cfg = OpenDoorEventCfg(
    reset_door_state = EventTermCfg(func=microwave.close, params={"percentage": 0.2}),
    reset_openable_object_pose = EventTermCfg(func=set_object_pose, ...)
)

# 评估指标:
metrics = [
    SuccessRateMetric(),                          # 成功率
    DoorMovedRateMetric(microwave, reset_openness)  # 门移动速率
]
```

**使用示例**:

```python
task = OpenDoorTask(
    microwave,                   # ← 必须是 Openable 实例
    openness_threshold=0.8,      # 门打开 80% → success=True
    reset_openness=0.2,          # 重置时关到 20%
    episode_length_s=5.0,        # 5 秒超时
)
```

**任务执行流程**:

```
Episode 开始
  ↓
Events: reset_all + reset_door_state(percentage=0.2) + reset_pose
  ↓
Robot 执行动作 (PINK IK EEF 控制)
  ↓
每步检查:
  ├─ time_out?        → terminated=True
  └─ microwave.is_open(threshold=0.8)? → success=True, terminated=True
  ↓
Episode 结束，重置
```

---

## 六、官方示例代码详细运行过程

### 示例: `Gr1OpenMicrowaveEnvironment` (gr1_open_microwave_environment.py)

**完整调用链** (逐行注释):

```python
# ── 第 1 步: 创建环境类 ──
env_class = Gr1OpenMicrowaveEnvironment()  # 继承 ExampleEnvironmentBase
    # __init__ 内部:
    #   self.asset_registry = AssetRegistry()    ← 自动注册所有资产
    #   self.device_registry = DeviceRegistry()  ← 自动注册遥操作设备

# ── 第 2 步: 构建 CLI 参数 ──
arena_args = argparse.Namespace(
    headless=True,        # 无头模式 (不显示窗口)
    enable_cameras=True,  # 启用相机渲染
    device="cuda:0",      # GPU 设备
    num_envs=1,           # 并行环境数
    disable_fabric=True,  # 禁用 Fabric (单环境无意义)
    seed=42,
    embodiment="gr1_pink",  # ← G1 的 IK 控制器
    object=None,             # 无额外物体
    teleop_device=None,      # 无遥操作设备
    mimic=False,             # 非 Mimic 模式
)

# ── 第 3 步: 构建环境 ──
isaaclab_env = env_class.get_env(arena_args)
    # get_env() 内部:
    # ① 从 AssetRegistry 获取并实例化资产
    background = self.asset_registry.get_asset_by_name("kitchen")()
        # → KitchenBackground 实例
        # → usd_path = "https://omniverse-content-.../kitchen_background.usd"
    microwave  = self.asset_registry.get_asset_by_name("microwave")()
        # → Microwave 实例 (Openable)
        # → usd_path = "/home/rq/.cache/lightwheel_sdk/.../Microwave039.usd"
    embodiment = self.asset_registry.get_asset_by_name(args.embodiment)(enable_cameras=True)
        # → GR1T2PinkEmbodiment 实例
        # → 配置 PINK IK 控制器 + 重力补偿

    # ② 设定初始位姿
    embodiment.set_initial_pose(Pose(
        position_xyz=(-0.4, 0.0, 0.0),    # 机器人站在微波炉左侧 40cm 处
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),  # 面向微波炉
    ))
    microwave.set_initial_pose(Pose(
        position_xyz=(0.4, -0.00586, 0.22773),  # 放在操作台上
        rotation_wxyz=(0.7071068, 0, 0, -0.7071068),  # 门朝机器人
    ))

    # ③ 组合场景
    assets = [background, microwave]
    scene = Scene(assets=assets)

    # ④ 创建任务
    task = OpenDoorTask(
        microwave,
        openness_threshold=0.8,
        reset_openness=0.2,
        episode_length_s=5.0,
    )

    # ⑤ 创建并返回环境
    return IsaacLabArenaEnvironment(
        name="gr1_open_microwave",
        embodiment=embodiment,
        scene=scene,
        task=task,
    )

# ── 第 4 步: 构建器编排 ──
builder = ArenaEnvBuilder(isaaclab_env, arena_args)
builder.orchestrate()
    # orchestrator 协调资产间的交互:
    #   - 碰撞组分配 (kitchen→0, microwave→0, robot→0)
    #   - 激活 contact sensors (microwave)
    #   - 注册自定义观测函数 (get_all_robot_link_state, get_eef_*)
    #   - 注册自定义事件 (set_object_pose, microwave.close)
    #   - 注册终止条件 (microwave.is_open)

cfg = builder.compose_manager_cfg()
    # 生成 IsaacLabArenaManagerBasedRLEnvCfg:
    #   - scene_cfg:     kitchen (Asset) + microwave (Articulation) + robot (Articulation)
    #   - observations:   Policy (8 个观测项) + Camera (robot_pov_cam_rgb)
    #   - actions:       PinkInverseKinematicsActionCfg
    #   - events:        reset_all + reset_door_state + reset_pose
    #   - terminations:  time_out + success(door_openness≥0.8)
    #   - metrics:       SuccessRate + DoorMovedRate

# ── 第 5 步: 创建仿真环境 ──
cfg.observations.policy.actions = None  # ★ 修复: 移除 last_action (0 维错误)
cfg.scene.num_envs = 1                  # ★ 修复: 覆盖默认 4096

env = ManagerBasedRLEnv(cfg=cfg)
    # ManagerBasedRLEnv.__init__():
    #   ① 解析 scene_cfg → 在 GPU 上构建场景
    #   ② 初始化 Observation Manager (8 个 policy 观测项 + 1 个 camera)
    #   ③ 初始化 Action Manager (PINK IK)
    #   ④ 初始化 Event Manager (reset 逻辑)
    #   ⑤ 初始化 Termination Manager (time_out + success)

# ── 第 6 步: 运行仿真 ──
obs, _ = env.reset()  # 重置: 关微波炉门 → 初始位姿
for step in range(max_steps):
    action = env.action_space.sample()    # 随机采样 (或 VLA 预测)
    obs, reward, term, trunc, info = env.step(action)
    print(f"reward={reward}, terminated={term}")
    if term or trunc:
        break
env.close()
```

### 完整时序要点

| 阶段 | 引擎/组件 | 耗时 (估计) |
|---|---|---|
| AppLauncher 启动 | Kit + PhysX + 渲染 | 60-90s |
| Asset 加载 (kitchen US) | omniverse S3 CDN | 5-10s |
| Microwave 资产 | lightwheel_sdk 本地缓存 | < 1s |
| PINK IK 初始化 | pinocchio 库 | < 1s |
| 场景克隆到 Fabric | GPU 复制 | 1-2s |
| 单步 step | PhysX + IK + 渲染 | 10-50ms |
| 重置 event | 关节复位 + 门关闭 | < 10ms |

---

## 七、归一化系统

### 7.1 数据流

```
机器人动作 (传感器空间)           VLA 动作 (模型空间)
        │                                  ▲
        │ decode_action()                   │
        │  → unnormalize                    │
        │  → joint_limit clamp              │
        │                                  │
        ▼                                  │
  ┌──────────┐                      ┌──────────┐
  │  机器人   │                      │ VLA 模型  │
  └──────────┘                      └──────────┘
        │                                  ▲
        │ encode_state()                   │
        │  → normalize                     │
        ▼                                  │
机器人状态 (传感器空间)            VLA 状态 (模型空间)
```

### 7.2 归一化统计格式 (dataset_statistics.json)

```json
{
  "action": {
    "q01": [...], "q99": [...], "mask": [...]
  },
  "proprio": {
    "q01": [...], "q99": [...], "mask": [...]
  }
}
```

- `mask: bool` — True=归一化，False=原样透传
- `q01/q99` — 用于 `bounds_q99` 归一化类型

---

## 八、Sim-to-Real 域随机化

### 8.1 配置与 API

```python
from arena.config import Sim2RealConfig
from arena.sim2real import Sim2RealCurriculum, Randomizer

curriculum = Sim2RealCurriculum(Sim2RealConfig(level=2, seed=42))

# 观测侧: 光照(L) + 纹理(T) + 相机(C) + 本体噪声(mu)
randomized_obs = curriculum.randomize_observation(observation)

# 动力学侧(M): 采样物理参数缩放系数, 交给仿真器施加
dynamics = curriculum.randomize_dynamics()
# DynamicsRandomization(mass_scale, friction_scale, actuator_gain_scale, damping_scale)

# 延迟(N)
action = curriculum.delay_action(action)
```

> 也可以用 `Randomizer(CURRICULUM[level], seed=...)` 直接构造, 或调用更细粒度的
> `randomize_image(img, photometry)` / `randomize_camera(img, camera)` /
> `randomize_state(state)`。

### 8.2 四级课程详情（与 `arena/sim2real.py::CURRICULUM` 逐项一致）

每个量都是**最大相对偏差** `m`，而非绝对物理量：

| Level | 光照 L | 纹理 T | 相机 C | 动力学 M | 观测噪声 mu | 动作延迟 N |
|---|---|---|---|---|---|---|
| **DR-L1** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| **DR-L2** | 0.15 | 0.15 | 0.05 | 0.10 | 0.01 | 1 |
| **DR-L3** | 0.30 | 0.30 | 0.15 | 0.25 | 0.03 | 2 |
| **DR-L4** | 0.50 | 0.50 | 0.30 | 0.50 | 0.05 | 3 |

各量的采样方式：

```
L 光照    亮度乘子 = 1 + Unif(-m, m)
T 纹理    对比度乘子 = 1 + Unif(-m, m)
C 相机    zoom        = 1 + Unif(0, m)      # 只放大不缩小, 故不产生黑边
          shift_x/y   = Unif(-m, m)         # 归一化平移
          noise_sigma = m × 8               # 灰度级 (m=0.15 -> 1.2)
M 动力学  mass / friction / actuator_gain / damping = 1 + Unif(-m, m), 裁剪到 [0.5, 1.5]
mu 噪声   proprio += N(0, m)                # 在适配器归一化后的 [-1,1] 空间, 非物理单位
N 延迟    固定延迟 m 个控制步
```

> **DR-L1 是严格恒等变换**：所有量均为 0，`randomize_observation()` 原样返回调用方的数组
> （连数组对象身份都保持）。这样"标准仿真闭环"才是可复现的基线，也才能真正作为
> "先验证软件闭环"的第一级。

### 8.3 六个因子的实现边界（诚实说明）

`E ~ P(L, T, C, M, mu, N)` 中，本模块的覆盖情况：

| 因子 | 是否已实现 | 说明 |
|---|---|---|
| `L` 光照 | ✅ | `randomize_image` 亮度乘子 |
| `T` 纹理 | ✅ | `randomize_image` 对比度乘子 |
| `C` 相机 | ✅ | `randomize_camera`：裁剪 + 平移 + 传感器噪声（图像空间） |
| `mu` 观测噪声 | ✅ | `randomize_state` |
| `N` 延迟 | ✅ | `delay_action`（带缓冲区） |
| `M` 动力学 | ⚠️ **只产生参数** | `randomize_dynamics()` 返回缩放系数；**施加需由仿真器完成** |

**为什么 `M` 只产生参数**：`Randomizer` 刻意不依赖任何物理引擎，因此它能随机化
"图像字节"，但不能写物理属性。施加方式（以 Isaac Lab 为例）已写在
`DynamicsRandomization` 的 docstring 中：

```python
d = curriculum.randomize_dynamics()          # 每个 episode 采样一次
view = scene[asset_name].root_physx_view
view.set_masses(view.get_masses() * d.mass_scale, indices)
view.set_material_properties(
    view.get_material_properties() * [1.0, d.friction_scale, 1.0], indices
)
```

> **另需注意**：`C` 的随机化是**图像空间近似**（裁剪/平移/噪声），可以模拟焦距与相机
> 姿态变化对画面的影响，但**无法复现真正的透视/视差变化**——那必须由渲染器完成。
> 这样做是为了让 `arena.sim2real` 保持零依赖，可包裹任意具身（Isaac Lab / MuJoCo / 回放缓冲）。

---

## 九、CLI 命令速查

### 9.1 ARENA CLI

```bash
# 策略服务器 (GPU)
arena server --backend unifolm_vla \
  --ckpt_path unifolm-vla/models/UnifoLM-VLA-Base1/checkpoints/pytorch_model.pt \
  --port 8777

# 仿真闭环 (Mock 后端)
arena sim --backend mock --instruction "open the microwave door"

# 仿真闭环 (VLA 后端)
arena sim --backend http --server_url http://127.0.0.1:8777/act

# 真机控制
arena real --server_url http://127.0.0.1:8777/act --instruction "pick up the cup"
```

**真实探活（推荐在编排/监控中使用）**:

```bash
# 健康 -> 200, 不健康 -> 503; 两者都返回结构化 JSON
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8777/health
curl -s http://127.0.0.1:8777/health | python -m json.tool
# {"status":"ok","healthy":true,"version":"0.1.0","backend":"MockPolicyBackend","detail":null}

# 官方 UnifoLM-VLA 部署服务器同样提供 /health
curl -s http://127.0.0.1:8777/health
```

### 9.2 G1-D 微波炉脚本

```bash
cd /home/rq/文档/project/arena_0_1_0

# Mock 后端 (无需 GPU/VLA 服务器)
python arena_g1_microwave.py --backend mock --max_steps 50

# HTTP 后端 (VLA 服务器已启动)
python arena_g1_microwave.py --backend http --max_steps 200

# 自定义指令
python arena_g1_microwave.py --backend mock \
  --instruction "open the microwave door gently" --max_steps 100

# 测试 Isaac Lab 通用环境
python arena_isaaclab_demo.py --env Isaac-Velocity-Flat-G1-v0 --backend mock
```

### 9.3 框架演示

```bash
# ARENA 端到端演示 (无需 GPU)
python demo.py
python demo.py --episodes 2 --sim2real-level 3

# 单元测试
/home/rq/miniconda3/envs/unifolm-vla/bin/python -m pytest tests/ -q
```

---

## 十、集成脚本：G1-D 厨房微波炉

### 脚本: `arena_g1_microwave.py` (基于三核心模块)

**设计**:

```
main()
  ├─ AppLauncher               ← 1. 启动 Isaac Sim
  ├─ AssetRegistry             ← 2. 注册资产
  ├─ Scene 构建                ← 3. Scene 模块
  │    ├─ kitchen (背景)
  │    ├─ microwave (物体 → Openable)
  │    └─ gr1_pink  (GR1T2 + PINK IK)  ← Embodiment 模块
  ├─ OpenDoorTask              ← 4. Task 模块
  ├─ IsaacLabArenaEnvironment  ← 5. 组合
  ├─ ArenaEnvBuilder           ← 6. 生成配置
  ├─ ManagerBasedRLEnv(cfg)   ← 7. 直接实例化
  ├─ G1MicrowaveRobot(env)     ← 8. ARENA RobotInterface 包装
  └─ ControlLoop.run()         ← 9. 闭环执行
```

**环境变量要求**:

| 环境变量 | 说明 |
|---|---|
| `CONDA_PREFIX` | env_isaaclab 环境 |
| `LD_LIBRARY_PATH` | Isaac Sim 库路径 (自动配置) |

**运行流程** (100 步示例):

```
2026-09-17 07:02:15 [INFO] Launching Isaac Sim (headless=True, cameras=True)...
2026-09-17 07:02:45 [INFO] Loading GR1T2 Pink embodiment...
2026-09-17 07:02:55 [INFO] Loading kitchen background...
2026-09-17 07:02:56 [INFO] Loading microwave object...
2026-09-17 07:02:56 [INFO] Task: OpenDoorTask
2026-09-17 07:03:47 [INFO] G1 Microwave environment created — action_dim=36
2026-09-17 07:03:47 [INFO] Starting control loop...
...
2026-09-17 07:04:15 [INFO] Episode complete!
```

---

## 十一、G1 移动操作箱子抓取与放置 (Loco-Manipulation Pick & Place)

### 11.1 任务概述

G1 移动操作箱子抓取与放置任务 (G1 Loco-Manipulation Box Pick and Place Task)
是指宇树科技 G1 人形机器人结合下肢移动 (Locomotion) 与双臂操作 (Manipulation)
来完成箱子抓取并放置到目标分拣箱的任务。

**核心词汇**:
- **G1**: 宇树科技 (Unitree) 研发的通用人形机器人
- **Loco-Manipulation (移动操作)** : 机器人边移动边进行手臂抓取和操作
- **Pick and Place (抓取与放置)** : 识别、夹取物体并放置到目标位置

### 11.2 任务流程 (4 个阶段)

```
阶段 1: 导航至操作台 (navigate_to_table)
  G1 从初始位置步行移动到放有棕色箱子的操作台前
  导航子目标: [0.18, 0.18, 0.0] → 直行 18cm 到箱子面前
  ─────────────────────────────────────────
阶段 2: 转向操作台 (turn_in_place)
  G1 在操作台面前原地转身约 102° 面朝箱子
  导航子目标: [0.18, 0.18, -1.78] → 保持位置，旋转朝向箱子
  ─────────────────────────────────────────
阶段 3: 导航至分拣箱 (navigate_to_bin)
  G1 抓住箱子后步行移动到蓝色分拣箱位置
  导航子目标: [-0.0955, -1.107, -1.78] → 移动约 1.3m 到分拣箱前
  ─────────────────────────────────────────
阶段 4: 放置箱子 + 成功检测
  G1 松开箱子，检测箱子是否在分拣箱范围内
  成功条件: 箱子在分拣箱 26×13×15cm 接近范围内
```

### 11.3 场景配置

| 资产 | 名称 | 位置 | 说明 |
|------|------|------|------|
| 背景 | `galileo_locomanip` | — | 开放式操作空间（地面 + 操作台 + 围栏 + 货架） |
| 抓取物体 | `brown_box` | (0.5785, 0.18, 0.0707) | 棕色箱子，放在操作台上 |
| 目标容器 | `blue_sorting_bin` | (-0.245, -1.627, -0.264) | 蓝色分拣箱，放在地面上 |
| 机器人 | `g1_wbc_pink` | (0.0, 0.18, 0.0) | 宇树 G1 + 全身控制 |

### 11.4 实施例: 宇树 G1 + WBC 全身控制

**`g1_wbc_pink`** 是 G1 专用实施例，启用全身控制 (Whole Body Control):

```
G1 WBC Pink Embodiment
  ├─ 下肢: 6 DOF × 2 = 12 维 (髋/膝/踝关节)
  ├─ 上肢: 7 DOF × 2 = 14 维 (PINK IK 控制)
  ├─ 手部: 手指关节若干维
  ├─ 导航: P-Controller 步行轨迹跟踪
  │    ├─ navigate_to_table   → [0.18, 0.18, 0.0]
  │    ├─ turn_in_place       → [0.18, 0.18, -1.78]
  │    └─ navigate_to_bin     → [-0.0955, -1.107, -1.78]
  └─ 动作空间总计: ~50+ 维 (动态确定)
```

**与 GR1T2 Pink 的对比**:

| 特性 | GR1T2 Pink (微波炉) | G1 WBC Pink (移动操作) |
|------|----------------------|--------------------------|
| 下肢控制 | 固定位置 (无行走) | **WBC 全身控制 (导航)** |
| 上肢 IK | PINK IK (14维) | PINK IK (14维) |
| 手部 | Fourier 手 (22维) | 宇树 G1 手 |
| episode_length | 5.0s | **20.0s** |
| action_dim (对外契约) | **36** | **23**（见 §12.2） |
| action_dim (env.action_space) | 36 | ~50+（动态确定） |
| 导航子目标 | 无 | **4 个 (含转向)** |

> ⚠️ **关于"~50+"与"23"的差异**：`env.action_space` 反映的是**仿真底层**的控制量
> （下肢 12 + 双臂 IK 14 + 手部 + 导航参数）；而 ARENA 对外暴露、也是策略实际推理所用的
> **规范动作契约是 23 维**（`DOCUMENTATION.md §12.2`、`arena_g1_locomanip_pnp_lerobot.py`）。
> 二者不可混用；完整对照见 §3.2「各实施例维度对照」。

### 11.5 任务逻辑: G1LocomanipPickAndPlaceTask

```python
class G1LocomanipPickAndPlaceTask(TaskBase):
    def __init__(self, pick_up_object, destination_bin, background_scene, episode_length_s=20.0)
```

**成功条件** — `objects_in_proximity`:

| 参数 | 值 | 含义 |
|------|-----|------|
| `max_x_separation` | 0.260 | 箱子 X 距离分拣箱 ≤ 26cm |
| `max_y_separation` | 0.130 | 箱子 Y 距离分拣箱 ≤ 13cm |
| `max_z_separation` | 0.150 | 箱子 Z 距离分拣箱 ≤ 15cm |

三轴同时满足 → `success=True`

**失败条件** — `root_height_below_minimum`:
- `minimum_height=-0.6` → 箱子中心 Z < -0.6m → `object_dropped=True`

**重置事件**:
- 箱子位姿随机化: 在初始位置 ±2.5cm 范围内均匀扰动

### 11.6 运行方法

```bash
conda activate env_isaaclab
cd /home/rq/文档/project/arena_0_1_0

# GUI 模式 (Mock 后端, 500 步)
python arena_g1_locomanip_pnp.py --no_headless --backend mock --max_steps 500

# Headless 模式
python arena_g1_locomanip_pnp.py --backend mock --max_steps 500

# 对接 VLA 服务器
python arena_g1_locomanip_pnp.py --no_headless --backend http --max_steps 500
```

### 11.7 代码架构 (`arena_g1_locomanip_pnp.py`)

```
main()
  ├─ AppLauncher                              ← 1. 启动 Isaac Sim
  ├─ AssetRegistry                            ← 2. 资产注册
  ├─ Scene 构建                               ← 3. Scene 模块
  │    ├─ galileo_locomanip (操作场景)
  │    ├─ brown_box (箱子, 操作台上)
  │    └─ blue_sorting_bin (分拣箱, 地面)
  ├─ g1_wbc_pink                              ← 4. Embodiment 模块
  │    (宇树 G1 + WBC + PINK IK + 导航)
  ├─ G1LocomanipPickAndPlaceTask              ← 5. Task 模块
  │    ├─ 成功: objects_in_proximity (26×13×15cm)
  │    └─ 失败: object_dropped (Z<-0.6m)
  ├─ IsaacLabArenaEnvironment + ArenaEnvBuilder  ← 6. 配置生成
  ├─ ManagerBasedRLEnv(cfg=cfg)               ← 7. 创建环境
  ├─ G1LocomanipRobot(env)                    ← 8. ARENA RobotInterface
  └─ ControlLoop.run()                        ← 9. 闭环执行
```

### 11.8 预期 GUI 视图

```
        ┌─────────────────────────────────────┐
        │          galileo_locomanip          │
        │                                     │
        │  [brown_box] @ (0.58, 0.18, 0.07)  │  ← 操作台
        │     ┌──┐                            │
        │     └──┘                            │
        │                                     │
        │  [G1]                               │
        │   @ (0, 0.18, 0)                   │  ← 初始位置
        │    ┌┴┐                              │
        │    │ │                              │
        │    └┬┘                              │
        │                                     │
        │            [blue_sorting_bin]       │  ← 目标
        │          @ (-0.25, -1.63, -0.26)   │
        │            ┌─────┐                  │
        │            └─────┘                  │
        └─────────────────────────────────────┘

任务序列:
  G1 步行到操作台 → 转身 → 抓取箱子 → 步行到分拣箱 → 放置
```

---

---

## 十二、LeRobot 兼容封装 — G1 移动操作 PnP

> **文件**: `arena_g1_locomanip_pnp_lerobot.py`  
> **目的**: 将独立仿真实验包装为 LeRobot 生态兼容的 Gymnasium 环境，支持 `lerobot-eval` 命令行评估 VLA 策略（如 SmolVLA）。

### 12.1 架构层级（5 层）

```
第 0 层: Monkey-Patch 栈 (自动应用)
  ├── Patch 1/4: URDF 重定向 (Nucleus → GitHub Unitree 本地克隆)
  ├── Patch 2/4: WBC 策略 Stub (G1HomiePolicyV2 → _g1_wbc_stub.py)
  ├── Patch 3/4: PINK IK 求解器 (osqp → quadprog, 双重 monkey-patch)
  └── Patch 4/4: Mock 四元数归一化 (InProcessClient 内安全修复)

第 1 层: Isaac Sim 引擎
  └── AppLauncher(headless, enable_cameras=False)

第 2 层: Arena 三核心
  ├── Scene: galileo_locomanip + brown_box + blue_sorting_bin
  ├── Embodiment: g1_wbc_pink (Unitree G1 + WBC Withen Body Control)
  └── Task: G1LocomanipPickAndPlaceTask (episode_length=20s)

第 3 层: ArenaEnvBuilder → ManagerBasedRLEnv
  └── orchestrate() + compose_manager_cfg() + 3 项配置修复

第 4 层: IsaacLabEnvWrapper → Gymnasium AsyncVectorEnv
  ├── reset() → (obs, info) — LeRobot Gymnasium 标准格式
  ├── step(action) → (obs, reward, terminated, truncated, info)
  └── _get_success() — 从 TerminationManager 读取 success 标志

第 5 层: 使用接口
  ├── 方法 A: 直接 Python 调用
  │   python arena_g1_locomanip_pnp_lerobot.py --success_test
  │   python arena_g1_locomanip_pnp_lerobot.py --run_loop --max_steps 100
  └── 方法 B: lerobot-eval (需要 VLA 策略 ONNX 模型)
```

### 12.2 动作空间 (23 维)

| 索引 | 名称 | 维度 | 说明 |
|------|------|------|------|
| 0-1 | if_navigate / nav_vel | 2 | 导航命令 |
| 2-4 | left_eef_pos | 3 | 左手末端执行器位置 |
| 5-8 | left_eef_quat | 4 | 左手末端执行器四元数 [w,x,y,z] |
| 9-11 | right_eef_pos | 3 | 右手末端执行器位置 |
| 12-15 | right_eef_quat | 4 | 右手末端执行器四元数 [w,x,y,z] |
| 16-18 | navigate_cmd | 3 | 导航子目标 (x, y, heading) |
| 19 | base_height_cmd | 1 | 底座高度命令 |
| 20-22 | torso_orientation_cmd | 3 | 躯干姿态 RPY |

### 12.3 观测空间 (policy 组)

| 索引 | 名称 | 形状 | 说明 |
|------|------|------|------|
| 0 | robot_joint_pos | (43,) | 全身关节位置 |
| 1 | robot_joint_vel | (43,) | 全身关节速度 |
| 2 | right_wrist_pose_pelvis_frame | (4, 4) | 右腕位姿 (pelvis 坐标系) |
| 3 | left_wrist_pose_pelvis_frame | (4, 4) | 左腕位姿 (pelvis 坐标系) |
| 4 | left_eef_pos | (3,) | 左手末端执行器位置 |
| 5 | left_eef_quat | (4,) | 左手末端执行器四元数 |
| 6 | right_eef_pos | (3,) | 右手末端执行器位置 |
| 7 | right_eef_quat | (4,) | 右手末端执行器四元数 |
| 8 | body_eef_pos | (3,) | 躯干位置 |
| 9 | body_eef_quat | (4,) | 躯干四元数 |
| 10 | robot_pos | (3,) | 机器人世界位置 |
| 11 | robot_quat | (4,) | 机器人世界四元数 |

### 12.4 成功/失败条件

| 条件 | 类型 | 说明 |
|------|------|------|
| **success** | `objects_in_proximity` | 棕箱与分拣箱距离 ΔX<26cm, ΔY<13cm, ΔZ<15cm |
| **object_dropped** | `root_height_below_minimum` | 箱子 Z < -0.6m |
| **time_out** | `mdp_isaac_lab.time_out` | Episode 超过 20 秒 |

### 12.5 运行命令

```bash
# 激活环境
conda activate env_isaaclab
cd /home/rq/文档/project/arena_0_1_0

# 方式 A: success_test 模式 (验证 PnP 终止条件)
python arena_g1_locomanip_pnp_lerobot.py --success_test

# 方式 B: Mock 策略闭环比环
python arena_g1_locomanip_pnp_lerobot.py --no_headless --run_loop --max_steps 200

# 方式 C: 通过 lerobot-eval 评估 VLA 策略
lerobot-eval \
    --env.type=isaaclab_arena \
    --env.hub_path=nvidia/isaaclab-arena-envs \
    --env.environment=g1_locomanip_pnp \
    --env.embodiment=g1_wbc_pink \
    --env.object=brown_box \
    --env.headless=false \
    --policy.path=nvidia/smolvla-arena-g1-locomanip-pnp \
    --eval.batch_size=1
```

### 12.6 为什么正常模式下 success: false？

| 层 | 当前运行 | 期望行为 |
|----|----------|----------|
| WBC 策略 | `_g1_wbc_stub.py` → 固定站立 | ONNX 模型 → 全身协调运动 |
| Mock 后端 | 23 维随机动作 | VLA 策略 → 4 阶段子目标序列 |
| 导航 | 随机 `navigate_cmd` | P-Controller 驱动步行 |

**视觉表现**: Stub 固定腿部 → 随机手臂碰到箱子 → IK 求解随机姿态 → 重心不稳 → "趴在架子上"。

### 12.7 四层 Monkey-Patch 技术细节

| 层 | 缺陷 | 修复 | 位置 |
|----|------|------|------|
| 1 | Nucleus URDF 不可达 | `retrieve_file_path` monkey-patch → GitHub Unitree | `_patch_assets()` |
| 2 | ONNX 模型不可达 | Stub 类替换 `G1HomiePolicyV2` | `_patch_wbc_policy()` |
| 3 | qpsolvers 4.x `osqp` 缺失 | 双重 patch: 源文件 + 运行时 | `_patch_pink_ik()` + `_patch_source_files()` |
| 4 | Mock 随机四元数零范数 | `InProcessClient.infer()` 归一化 | `_patch_quaternion()` + `build_mock_client()` |

### 12.8 文件结构

```
arena_0_1_0/
├── arena_g1_locomanip_pnp.py          # 独立仿真实验 (arena 框架)
├── arena_g1_locomanip_pnp_lerobot.py  # ★ LeRobot 兼容封装 (Gymnasium)
├── _g1_wbc_stub.py                    # WBC 策略 Stub (共享)
├── arena/                             # ARENA 框架
├── IsaacLab/                          # Isaac Lab 基础设施
│   └── isaaclab-arena-envs/           # LeRobot EnvHub 环境定义
│       ├── env.py                     # LeRobot 入口
│       ├── isaaclab_env_wrapper.py    # Gymnasium 适配器
│       └── example_envs.yaml          # 5 个可用环境
└── DOCUMENTATION.md                   # 本文档
```

---

> **文档版本**: 4.0  
> **生成日期**: 2026-09-18  
> **覆盖范围**: arena 0.1.0 / Isaac Lab Arena SDK / Isaac Lab 核心 / UnifoLM-VLA / G1 移动操作抓取放置 / LeRobot 兼容封装
