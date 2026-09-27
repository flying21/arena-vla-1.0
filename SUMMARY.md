# ARENA 0.1.0 — 工作成果总结与文件清单

对应技术报告《VLA Server–Client–Adapter（AI+ | Vision–Language–Action | Isaac Lab / Arena | Unitree）》。

## 一、总体目标

实现将 GPU 上的 VLA 策略服务器与 Isaac Lab/Arena 仿真、Unitree 真机打通的三层框架：

```
VLA Policy Server (GPU)  ──HTTP──►  Client  ──►  Robot / Simulation
                                      │
                                Embodiment Adapter
```

核心公式（报告 §2.1 / §2.2）：

- 观测：`O_t = { I_head, I_wrist, S_t, L }`
- 动作块：`A_{t:t+H} = [a_t, a_{t+1}, ..., a_{t+H}]`

## 二、新增文件清单

### 2.1 核心库 `arena/`

| 文件 | 说明 | 报告章节 |
|---|---|---|
| `arena/__init__.py` | 包入口，导出核心类型与配置 | — |
| `arena/types.py` | 规范观测 `Observation` / 动作块 `ActionChunk`（含中文注释） | §2.1 / §2.2 |
| `arena/config.py` | 配置数据类 + 归一化/反归一化工具（含中文注释） | §4.4 |
| `arena/adapter.py` | Embodiment Adapter（观测适配器 + 动作适配器） | §4.1–4.3 |
| `arena/backends.py` | 可插拔策略后端（mock/unifolm_vla/openpi/http） | §2.3 / §2.4 |
| `arena/server.py` | FastAPI 策略服务器（`/act`、`/health`） | §2.3 |
| `arena/client.py` | 策略客户端 + 仿真/真机控制回路 | §3.1–3.3 |
| `arena/sim2real.py` | Sim-to-Real 域随机化课程（Level 1–4） | §5.1 / §5.2 |
| `arena/cli.py` | 命令行入口 `arena server/sim/real` | — |
| `arena/configs/default.yaml` | 默认配置文件 | — |

### 2.2 项目根目录

| 文件 | 说明 |
|---|---|
| `pyproject.toml` | 打包配置，注册 `arena` 命令，声明依赖与可选依赖 |
| `demo.py` | 端到端演示（mock 后端 + 内存机器人，无需 GPU） |
| `README.md` | 快速上手与架构说明（英文） |
| `SUMMARY.md` | 本文件：工作总结与文件清单 |
| `DOCUMENTATION.md` | 技术细节文档（环境/模型/数据/调用关系） |
| `tests/test_arena.py` | 单元测试 |

### 2.3 复用（未修改）的既有资源

| 路径 | 用途 |
|---|---|
| `unifolm-vla/` | VLA 训练/推理代码与模型配置 |
| `unifolm-vla/deployment/model_server/run_real_eval_server.py` | `UnifoLMVLABackend` 实现蓝本 |
| `unifolm-vla/experiments/LIBERO/` | 仿真评测流程参考 |
| `unifolm-vla/arena_test.py` | LeRobot Arena 环境接入参考 |
| `IsaacLab/` | 仿真环境基础设施 |

## 三、修改文件清单

本项目为**新增实现**，未修改 `unifolm-vla/` 与 `IsaacLab/` 下的任何既有文件，
仅在项目根目录与新建的 `arena/` 包内新增代码，既有仓库保持原样。

## 四、模块职责一览

| 模块 | 关键类/函数 | 职责 |
|---|---|---|
| `arena.types` | `Observation`, `ActionChunk` | 统一数据结构 |
| `arena.config` | `ArenaConfig`, `normalize`, `unnormalize` | 参数管理与归一化 |
| `arena.adapter` | `EmbodimentAdapter` 等 | 机器人 ↔ VLA 双向转换 |
| `arena.backends` | 4 种后端 + `health()` / `status()` | 策略推理后端抽象 + **真实探活** |
| `arena.server` | `PolicyServer` | HTTP 推理服务（`/act`、`/health` 200/503） |
| `arena.client` | `PolicyClient`, `ControlLoop` | 客户端与统一闭环 |
| `arena.sim2real` | `Sim2RealCurriculum`, `CameraRandomization`, `DynamicsRandomization` | 域随机化课程（L/T/C/μ/N 直接施加，M 输出参数） |
| `arena.cli` | `main` | 命令行入口 |

## 五、快速验证

```bash
cd /home/rq/文档/project/arena_0_1_0
/home/rq/miniconda3/envs/unifolm-vla/bin/python -c "import arena; print(arena.__version__)"
/home/rq/miniconda3/envs/unifolm-vla/bin/python demo.py
/home/rq/miniconda3/envs/unifolm-vla/bin/python -m pytest tests/ -q
```

详见 `DOCUMENTATION.md`。