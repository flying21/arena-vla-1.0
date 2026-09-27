# Ubuntu 22.04 安装指南：Isaac Sim 5.1 + Isaac Lab 2.3 + Isaac Lab Arena 0.1.0

> **文档目的**：在 Ubuntu 22.04 上搭出可用于 `arena_vla_1.0` 项目的 Isaac 仿真环境。
> **信息来源**：NVIDIA 官方文档站（Isaac Lab / Isaac Sim / Isaac Lab-Arena）与官方 GitHub 仓库，抓取日期见文末「参考来源」。凡属我的推断或需你自行确认的内容，均以 **⚠️ 需确认** 明确标注。
> **编写日期**：2026-09（依据官方文档当时快照）

---

## 0. 先读这一段：三个必须先知道的结论

### 结论 1：你要求的版本组合是被支持的，但官方两处文档自相矛盾

关于 **Arena 0.1.0 配哪个 Isaac Sim 版本**，NVIDIA 官方给出了**互相冲突**的两个说法：

| 来源 | Arena 版本 | Isaac Lab | **Isaac Sim** | Python |
|---|---|---|---|---|
| [Arena 0.1.0 安装页](https://isaac-sim.github.io/IsaacLab-Arena/release/0.1.0/pages/quickstart/installation.html)（版本专属文档） | 0.1.0 | 2.3.0 | **5.1.0** | ≥3.10 |
| [Arena README 兼容表](https://github.com/isaac-sim/IsaacLab-Arena)（`main` 分支，较新） | `release/0.1.0` | 2.3.0 | **5.0.0** | ≥3.10 |
| 同上 | `release/0.1.1` | 2.3.0 | 5.0.0 | ≥3.10 |
| 同上 | `feature/arena_v0.2_on_lab_2.3` | 2.3.0 | 5.1.0 | ≥3.10 |

Arena 0.1.0 安装页原文（逐字）：

> "Isaac Lab Arena runs on Isaac Sim `5.1.0` and Isaac Lab `2.3.0`. The dependencies are installed automatically during the Docker build process."

**本指南的取舍**：以 **Isaac Sim 5.1.0** 为准 —— 因为 0.1.0 的安装页是该版本专属文档，而 README 表格位于 `main` 分支、描述的是新版本谱系的回溯归类。**⚠️ 需确认**：若你按本指南装完发现资产加载或 API 不兼容，请回退到 Isaac Sim 5.0.0 验证是否是版本表那一侧正确。

### 结论 2：Isaac Sim 5.1.0 已是「不再支持」的版本

Isaac Sim 5.1.0 的官方需求页顶部带警告条（逐字）：

> "⚠ **Unsupported release:** Isaac Sim 5.1.0 is no longer supported. Bug fixes and new features are delivered only in newer releases."

也就是说这个组合能装、能用，但**不会再收到修复**。如果项目允许，考虑升级到 Arena 0.2.x + Isaac Sim 6.x（见 README 兼容表）。但 `arena_vla_1.0` 的代码是按 5.1 + Lab 2.3 写的（见第 8 节），所以本指南仍以你的要求为准。

### 结论 3：Arena 0.1.0 **只能通过 Docker 安装**

Arena 0.1.0 安装页原文（逐字）：

> "This page describes how to install Isaac Lab Arena from source inside a Docker container."
> "Isaac Lab Arena supports installation from source inside a Docker container. Future versions of Isaac Lab Arena, we will support a larger range of installation options."

Docker 容器页进一步说明：

> "This first version of Isaac Lab Arena is designed to run inside a Docker container."

所以 **路线 A（Docker）是 Arena 0.1.0 的唯一官方路径**；路线 B / C 只能装到 Isaac Sim + Isaac Lab，装不了 Arena 0.1.0。这一点决定了你的整体方案选择。

---

## 1. 硬件与系统要求

来源：[Isaac Sim 5.1.0 Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)（x86_64）

| 项目 | 最低 | 推荐 | 理想 |
|---|---|---|---|
| 操作系统 | Ubuntu 22.04 / 24.04 或 Windows 10/11 | 同左 | 同左 |
| CPU | Intel Core i7 第 7 代 / AMD Ryzen 5 | i7 第 9 代 / Ryzen 7 | i9 / Ryzen 9 / Threadripper |
| 核心数 | 4 | 8 | 16 |
| 内存 | 32 GB | 64 GB | 64 GB |
| 存储 | 50 GB SSD | 500 GB SSD | 1 TB NVMe SSD |
| GPU | GeForce RTX 4080 | GeForce RTX 5080 | RTX PRO 6000 Blackwell |
| 显存 | 16 GB | 16 GB | 48 GB |
| **驱动（Linux）** | **580.65.06** | 580.65.06 | 580.65.06 |

官方补充说明（要点）：

- **容器方式只支持 Linux**；
- 需要联网：Isaac Sim 资产在线拉取，部分扩展也需联网；
- **无 RT Core 的 GPU（A100、H100）不支持**；
- 显存不足会直接影响教程与基准测试，传感器多的流程尤其敏感；
- 官方建议：新 GPU 或驱动异常时，用 `.run` 安装包装 [Unix Driver Archive](https://www.nvidia.com/en-us/drivers/unix/) 里的 **Latest Production Branch** 驱动。

> **对 `arena_vla_1.0` 的补充提醒**：项目文档里提到实际使用 **RTX 4090（23 GB 显存）**，并要求"启动前至少剩余 8 GB 空闲显存"。这满足最低要求，但跑 G1 WBC + PIK IK + 多相机时显存会比较紧张（项目文档也记录了"僵尸进程占用 VRAM 需 `kill -9` 清理"）。

### 1.1 安装 NVIDIA 驱动（580.65.06）

```bash
# 查看当前驱动与 GPU
nvidia-smi

# Ubuntu 官方仓库方式（版本可能低于 580，注意核对）
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# 若仓库版本不够新，用官方 .run 安装包：
#   1) 从 https://www.nvidia.com/en-us/drivers/unix/ 下载 580.65.06
#   2) 关闭图形界面后安装
sudo systemctl isolate multi-user.target
sudo sh ./NVIDIA-Linux-x86_64-580.65.06.run
sudo systemctl isolate graphical.target

# 验证（应看到 Driver Version: 580.65.06）
nvidia-smi
```

> **⚠️ 注意**：驱动升级后必须重启，并确认 `nvidia-smi` 正常。Isaac Sim 对驱动版本较敏感，版本过低会直接启动失败。

---

## 2. 三条安装路线怎么选

| 路线 | 装什么 | 能否得到 Arena 0.1.0 | 适用场景 |
|---|---|---|---|
| **A. Docker 一体化**（★ 推荐） | 容器内已含 Isaac Sim + Isaac Lab + Arena | ✅ **是**（唯一支持） | 跑 Arena 环境、`arena_g1_*` 系列脚本 |
| B. pip（宿主机） | isaaclab[isaacsim] 2.3.2 + Isaac Sim 5.1 | ❌ 否 | 只要 Isaac Lab，不需要 Arena |
| C. 二进制 + 源码（宿主机） | Isaac Sim 二进制 + Isaac Lab 源码 | ❌ 否 | 需要改 Isaac Lab 源码 / 无 Docker |

**建议**：既然你的目标是 Arena 0.1.0，直接用**路线 A**。路线 B/C 仅在你需要宿主机原生调试 Isaac Lab 时作为补充。

---

## 3. 路线 A（推荐）：Docker 一体化安装

### 3.1 安装 Docker 与 NVIDIA Container Toolkit

```bash
# ---- Docker Engine ----
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 让当前用户免 sudo 使用 docker（需重新登录生效）
sudo usermod -aG docker $USER
newgrp docker

# ---- NVIDIA Container Toolkit ----
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**验证容器能用到 GPU**：

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu:22.04 nvidia-smi
```

应能打印与宿主机一致的 GPU 信息。若报错，检查 `nvidia-ctk` 是否配置成功、是否重启了 docker。

### 3.2 克隆 IsaacLab-Arena（release/0.1.0）

仓库提供 **SSH** 与 **HTTPS** 两种方式，官方文档给的是 SSH：

```bash
# 方式一：SSH（官方文档写法，需配置 GitHub SSH key）
git clone --branch release/0.1.0 --recurse-submodules \
  git@github.com:isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena

# 方式二：HTTPS（无 SSH key 时用这个）
git clone --branch release/0.1.0 --recurse-submodules \
  https://github.com/isaac-sim/IsaacLab-Arena.git
cd IsaacLab-Arena

# 官方步骤里的第二条命令：确保子模块（含 Isaac Lab）已初始化
git submodule update --init --recursive
```

> **`--recurse-submodules` 很关键**：Isaac Lab 是作为 **git submodule** 引入的（见 README 的 Project Structure 中 `submodules/` 一项）。不加这个参数会得到一个缺少 Isaac Lab 的空壳。

### 3.3 启动容器

```bash
# Base 容器（推荐用于开发，体积小）
./docker/run_docker.sh

# Base + GR00T 容器（用于 GR00T 策略后训练与评估；体积大很多）
./docker/run_docker.sh -g
```

官方说明：

- `run_docker.sh` 会**先构建镜像再进入交互模式**；
- 0.1.0 提供两个容器：**Base**（Arena 代码 + 全部依赖）与 **Base + GR00T**（额外含 GR00T）；
- 官方建议：不确定用哪个就用 **Base**；
- **⚠️ GR00T 容器不支持 Blackwell GPU 与 DGX Spark**（官方原文明确）；
- 挂载目录（宿主 → 容器）：
  - `$HOME/datasets` → `/datasets`
  - `$HOME/models` → `/models`
  - `$HOME/eval` → `/eval`
- 这些目录可通过 `run_docker.sh` 的参数修改；完整参数列表见脚本 `docker/run_docker.sh`。

> **为什么挂载这三个目录很重要**：IsaacLab-Arena 会下载 USD 资产、数据集与预训练模型。挂到宿主后，**重启容器不用重新下载**。项目里的 G1 微波炉场景（`kitchen_background.usd`）与视频里的操作场景资产都走这条路径，首次拉取可能很慢。

建议先创建它们：

```bash
mkdir -p $HOME/datasets $HOME/models $HOME/eval
```

### 3.4 验证安装（官方给的两条 pytest）

容器内执行：

```bash
# 需要相机的测试
pytest -sv -m with_cameras isaaclab_arena/tests/ --ignore=isaaclab_arena/tests/policy/

# 不需要相机的测试
pytest -sv -m "not with_cameras" isaaclab_arena/tests/ --ignore=isaaclab_arena/tests/policy/
```

> **与 `arena_vla_1.0` 的关联**：项目文档记录了一个真实的踩坑 —— **Isaac Sim 5.1 的相机链路有 SyntheticData 缺陷**，导致 GUI 模式下相机数据为 0 维、必须 `enable_cameras=False`。这正是 `with_cameras` 这组测试值得单独跑的原因：它能提前暴露相机问题，而不用等到跑 `arena_g1_microwave.py` 时才发现拿到的是全黑占位图。

### 3.5 Omniverse 认证（访问内部资产时需要）

Arena 0.1.0 的部分资产当时仍托管在 NVIDIA 内部 Nucleus 上。安装页原文：

> "To allow the project to access Nvidia Omniverse assets or services (e.g., stored scenes, USD files, or extensions on a Nucleus server), you must authenticate using an Omniverse API token."

```bash
# OMNI_USER 必须是字面量字符串 "$omni-api-token"（含开头的美元符号）
export OMNI_USER='$omni-api-token'
export OMNI_PASS='<你生成的 API Token>'
```

Token 生成方式见 [Omniverse API Tokens 文档](https://docs.omniverse.nvidia.com/nucleus/latest/config-and-info/api_tokens.html)。

> **重要**：安装页的 TODO 注释写着 —— 该认证**仅用于尚未托管到公共 Nucleus 的资产，待公开后就不需要了**。所以：
> - 如果你能访问公共 Nucleus，这一段可以跳过；
> - 如果遇到 `omniverse://isaac-dev.ov.nvidia.com/...` 类地址访问失败，说明你需要的正是内部资产，要么配 Token，要么走下面的替代方案。

**替代方案（项目已实现）**：`arena_vla_1.0/arena_g1_locomanip_pnp.py` 里的 `_patch_g1_wbc_assets()` 把 Nucleus URL **重定向到本地克隆的 Unitree GitHub URDF**（`/tmp/unitree_ros_g1/robots/g1_description/`），并配套 `download_g1_wbc_assets.py` 做预下载。这是在无内部权限时让环境能构建起来的关键手段。

---

## 4. 路线 B：pip 安装 Isaac Lab 2.3.2 + Isaac Sim 5.1（宿主机，无 Arena）

> 仅在你不需要 Arena 时使用。**该路线装不出 Arena 0.1.0。**

来源：[Isaac Lab v2.3.2 — Installation using Isaac Lab Pip Packages](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/isaaclab_pip_installation.html)

### 4.1 创建 Python 环境

**Python 版本必须与 Isaac Sim 一致**，官方明确：

> "For Isaac Sim 5.X, the required Python version is **3.11**."
> "Using a different Python version will result in errors when running Isaac Lab."

```bash
# 用 Conda（官方推荐 Miniconda）
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab

# 或 venv
python3.11 -m venv env_isaaclab
source env_isaaclab/bin/activate

pip install --upgrade pip
```

### 4.2 安装依赖

```bash
# 同时装 Isaac Lab 扩展 + Isaac Sim
pip install isaaclab[isaacsim,all]==2.3.2 --extra-index-url https://pypi.nvidia.com

# 安装与 CUDA 12.8 匹配的 PyTorch（x86_64 Linux）
pip install -U torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128

# 若要用 rl_games 训练/推理，装其 Python 3.11 分支
pip install git+https://github.com/isaac-sim/rl_games.git@python3.11
```

> **版本说明**：Isaac Lab 2.3.2 安装页给的是 `==2.3.2`；而 `main` 分支文档给的是 `==2.3.2.post1`。**⚠️ 需确认**：请以你实际查阅的文档版本为准；两者都是 2.3.x 系列，但 `.post1` 是补丁版。若要严格对齐 2.3.0，把 `==2.3.2` 换成 `==2.3.0`。
>
> **aarch64（DGX Spark）不同**：torch 用 `2.9.0`/`torchvision 0.24.0` + `cu130`，且需要处理 `libgomp` 预加载告警（官方给了 `unset LD_PRELOAD` 的修法）。

### 4.3 验证

```bash
# 启动模拟器（首次运行会拉取扩展，可能超过 10 分钟）
isaacsim

# 指定 experience 文件
isaacsim isaacsim.exp.full.kit
```

**首次运行会要求接受 EULA**，官方给出的交互原文：

```
By installing or using Isaac Sim, I agree to the terms of NVIDIA OMNIVERSE LICENSE AGREEMENT (EULA)
in https://docs.isaacsim.omniverse.nvidia.com/latest/common/NVIDIA_Omniverse_License_Agreement.html

Do you accept the EULA? (Yes/No): Yes
```

> 无图形界面环境下可用 `export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y`（Arena 0.3.0 README 的写法；**⚠️ 需确认** 5.1 是否同样识别这两个变量）。

### 4.4 生成 VS Code 智能提示配置

```bash
python -m isaaclab --generate-vscode-settings
```

> 官方警告：该命令会生成 `.vscode/settings.json`，**若文件已存在会被覆盖**（会有确认提示）。项目仓库里已有一份 676 行的 Isaac Sim 模板（内容与本项目无关），执行前请先备份。

---

## 5. 路线 C：Isaac Sim 5.1 二进制 + Isaac Lab 源码（宿主机）

来源：[Isaac Lab v2.3.2 — Installation using Isaac Sim Pre-built Binaries](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html)

### 5.1 下载并解压 Isaac Sim 5.1 二进制

从 [Isaac Sim Download](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) 下载 **5.1.0** 的 zip，解压到 `${HOME}/isaacsim`。

```bash
# 解压（示例）
mkdir -p ${HOME}/isaacsim
unzip isaac-sim-5.1.0-linux-x86_64.zip -d ${HOME}/isaacsim

# 设置环境变量（建议写入 ~/.bashrc）
export ISAACSIM_PATH="${HOME}/isaacsim"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
```

> **注意**：上面 `unzip` 的具体文件名与内部目录结构**⚠️ 需确认** —— 官方示例只说明"解压到目标目录"并假设目录名为 `${HOME}/isaacsim`，未给出 zip 内的层级。请按 [Workstation Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html) 的示例操作，确保 `${ISAACSIM_PATH}/isaac-sim.sh` 存在。

**验证**：

```bash
# 启动模拟器
${ISAACSIM_PATH}/isaac-sim.sh

# 验证 python 路径
${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"

# 用独立脚本验证
${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.api/add_cubes.py
```

> **从旧版本升级时必做**（官方 Caution 原文）：
> ```bash
> ${ISAACSIM_PATH}/isaac-sim.sh --reset-user
> ```
> 用于清除旧的用户数据与缓存变量。不执行可能因为残留配置导致启动异常。

### 5.2 安装 Isaac Lab 2.3 源码

```bash
# 克隆（建议 fork 后替换 isaac-sim 为你的用户名）
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 切换到你需要的分支/标签（2.3 系列）
# ⚠️ 需确认：官方文档未直接给出 2.3 的 tag 名，请用 git tag -l 'v2.3*' 查看
git tag -l 'v2.3*'

# 建立 Isaac Sim 符号链接（关键步骤，让 Isaac Lab 能找到 Isaac Sim 的 python 模块与扩展）
ln -s ${ISAACSIM_PATH} _isaac_sim

# 安装系统依赖（robomimic 需要）
sudo apt install cmake build-essential
```

**创建 Python 环境（可选但推荐）**：

```bash
# Conda（默认环境名 env_isaaclab）
./isaaclab.sh --conda
conda activate env_isaaclab

# 或 uv（实验性；注意 uv venv 默认不含 pip，需 --seed）
./isaaclab.sh --uv
source ./env_isaaclab/bin/activate
```

**安装扩展**：

```bash
# 安装全部学习框架（rl_games, rsl_rl, sb3, skrl, robomimic）
./isaaclab.sh --install

# 或只装指定框架
./isaaclab.sh --install rl_games
# 可选值：all / rl_games / rsl_rl / sb3 / skrl / robomimic / none
```

**验证**：

```bash
# 用 isaaclab.sh 的 python（虚拟环境方式也可直接用 python）
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

预期：弹出模拟器窗口、显示黑色视口，说明安装成功。按 `Ctrl+C` 退出。

> **常见报错**：`ModuleNotFoundError: No module named 'isaacsim'`
> 官方给的修法：确认虚拟环境已激活，并执行 `source _isaac_sim/setup_conda_env.sh`（uv 环境同样适用）。

### 5.3 快速验证训练链路

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless
```

---

## 6. Arena 0.1.0 的第二种获取方式：作为子模块嵌入你的仓库

来源：[Installing IsaacLab-Arena in Your Repository](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/arena_in_your_repo/external_installation.html)

> **⚠️ 重要前提**：该页面的**基础镜像是 `nvcr.io/nvidia/isaac-sim:6.0.0`**（main 分支文档），即它描述的是 **Arena 0.3 + Isaac Sim 6.0** 的用法。用在 0.1.0 + 5.1 上时，镜像 tag 需相应改为 5.1 的镜像。**⚠️ 需确认** 5.1 对应的容器 tag 名。

官方推荐的集成模式（逐字要点）：

> "The recommended way to consume IsaacLab-Arena from an external project is to include it as an **unmodified git submodule** and extend it purely through its registration API — without editing any file inside the Arena source tree."

目录结构：

```
my_project/
├── submodules/
│   └── IsaacLab-Arena/          ← 未修改的 Arena 子模块
├── my_package/
│   ├── pyproject.toml
│   ├── isaaclab_arena_environments/
│   │   ├── __init__.py
│   │   └── my_environment.py    ← 自定义环境类
├── docker/
│   └── Dockerfile
└── .gitmodules
```

添加子模块（0.1.0 请指定分支）：

```bash
# 官方写法（main）
git submodule add git@github.com:isaac-sim/IsaacLab-Arena.git submodules/IsaacLab-Arena

# 若要 0.1.0
git submodule add -b release/0.1.0 git@github.com:isaac-sim/IsaacLab-Arena.git submodules/IsaacLab-Arena
```

Dockerfile 要点（官方示例，注意注释说明前提）：

```dockerfile
# 基础镜像必须已装 Isaac Sim
# e.g. FROM nvcr.io/nvidia/isaac-sim:6.0.0        ← 0.1.0+5.1 需改成对应 5.1 的 tag

# 镜像必须已装 Isaac Lab
# e.g. RUN /isaaclab/isaaclab.sh -i

# 把子模块拷进镜像并安装 Arena
COPY submodules/IsaacLab-Arena /opt/arena
RUN /isaac-sim/python.sh -m pip install -e /opt/arena

# 之后再装你自己的包（顺序不能反）
COPY my_package /workspace/my_package
RUN /isaac-sim/python.sh -m pip install -e /workspace/my_package
```

> 官方也说明：如果你有满足前提的**非 Docker 环境**，可以直接装进那个环境，不必用 Docker。但对 **0.1.0** 而言，安装页只描述了 Docker 路径，所以非 Docker 属于**未官方验证**的用法。

### 6.1 版本兼容总表（完整）

| Isaac Lab-Arena | Isaac Lab | Isaac Sim | Python |
|---|---|---|---|
| `main` | 3.0.0 | 6.0.0 | ≥ 3.12 |
| `release/0.3.0` | 3.0.0 | 6.0.0 | ≥ 3.12 |
| `release/0.2.1` | 3.0.0 | 6.0.0 | ≥ 3.12 |
| `release/0.2.0` | 3.0.0 | 6.0.0 | ≥ 3.12 |
| `feature/arena_v0.2_on_lab_2.3` | 2.3.0 | 5.1.0 | ≥ 3.10 |
| `release/0.1.1` | 2.3.0 | 5.0.0 | ≥ 3.10 |
| `release/0.1.0` | 2.3.0 | 5.0.0 | ≥ 3.10 |

> 再次强调第 0 节的冲突：0.1.0 的**安装页说 5.1.0**，本表说 5.0.0。

---

## 7. 安装后验证清单

按顺序勾选，任何一步失败都不要继续往下：

- [ ] `nvidia-smi` 显示驱动 **580.65.06**、显存 ≥16 GB
- [ ] `docker run --rm --runtime=nvidia --gpus all ubuntu:22.04 nvidia-smi` 能打印 GPU
- [ ] `$HOME/{datasets,models,eval}` 已创建（避免容器重启后重复下载资产）
- [ ] `git submodule update --init --recursive` 已执行，`submodules/` 下有 Isaac Lab
- [ ] `./docker/run_docker.sh` 能进入交互容器
- [ ] 容器内两条 `pytest` 命令通过（`with_cameras` 与 `not with_cameras`）
- [ ] 容器内能 `import isaaclab_arena`、`import isaaclab`
- [ ] （若需内部资产）`OMNI_USER` / `OMNI_PASS` 已设置，USD 能正常拉取

`arena_vla_1.0` 相关的额外验证：

- [ ] `python arena_isaaclab_demo.py --env Isaac-Cartpole-Direct-v0 --backend mock` 能跑通（最轻量的 Isaac Lab 接入检查）
- [ ] `python -c "from isaaclab_arena.assets.asset_registry import AssetRegistry; print(len(AssetRegistry().assets))"` 能列出资产
- [ ] `python arena_g1_microwave.py --backend mock --max_steps 50` 能创建环境

---

## 8. 与本工作区 `arena_vla_1.0` 的对接

装好环境后，请注意项目代码对环境的**具体假设**（这些是读代码得到的，不是官方文档）：

| 项目假设 | 出处 | 说明 |
|---|---|---|
| conda 环境名 `env_isaaclab` | 多个脚本 docstring | 与官方 `./isaaclab.sh --conda` 默认名一致 |
| `enable_cameras=False` | `arena_g1_microwave.py`、`arena_g1_locomanip_pnp.py` | 因 **Isaac Sim 5.1 相机数据 0 维缺陷**，注释里写明"无论 headless 还是 GUI 都会触发 syntheticdata 崩溃" |
| `cfg.observations.policy.actions = None` | 同上 | 规避 `last_action` 观测在首次 reset 时产生 0 维张量 |
| `cfg.scene.num_envs = 1` | 同上 | 覆盖 Arena 默认的 4096 并行环境 |
| `env.unwrapped.sim._app_control_on_stop_handle = None` | 同上 | 避免关闭时崩溃 |
| PINK IK 用 `quadprog` 而非 `osqp` | `arena_g1_locomanip_pnp*.py` | qpsolvers 4.x 不再内置 osqp，需双重 patch |
| Nucleus URL 重定向到 Unitree GitHub | `_patch_g1_wbc_assets()` | 内部资产不可达时的替代 |
| WBC 下肢用 Stub 固定站立 | `_g1_wbc_stub.py` | ONNX 权重不可达；**机器人不会行走**，故 Mock 下 `success=false` 属预期 |

> **⚠️ 与官方文档的版本矛盾**：项目 `DOCUMENTATION.md` 里写的是 "Python 3.11 + **Isaac Lab 0.47.2**"。这个版本号与 NVIDIA 官方 "Isaac Lab 2.3" 的编号体系不一致，**⚠️ 需确认**是笔误、还是指某个内部/特定构建。建议以官方 2.3.0 为准，并在装完后核对 `pip show isaaclab` 的实际版本。

---

## 9. 常见问题与排错

| 症状 | 可能原因 | 处理 |
|---|---|---|
| `nvidia-smi` 失败或驱动版本过低 | 驱动 < 580.65.06 | 装 580.65.06；升级后重启 |
| 容器内看不到 GPU | 未装/未配置 NVIDIA Container Toolkit | `nvidia-ctk runtime configure --runtime=docker` 后重启 docker |
| `ModuleNotFoundError: No module named 'isaacsim'` | 虚拟环境未激活 或 未 source 环境脚本 | `source _isaac_sim/setup_conda_env.sh`（uv 同样适用） |
| 首次启动卡很久 | 正在从 registry 拉扩展 | 官方说明：首次可能**超过 10 分钟**，之后走缓存 |
| 启动即崩 | 升级自旧版本，残留用户数据 | `${ISAACSIM_PATH}/isaac-sim.sh --reset-user` |
| 相机数据为 0 维 / 图像全黑 | Isaac Sim 5.1 SyntheticData 缺陷 | 项目已用 `enable_cameras=False` 规避；或在容器内跑 `-m with_cameras` 测试定位 |
| `omniverse://isaac-dev.ov.nvidia.com/...` 访问失败 | 该地址是 NVIDIA 内部服务器 | 配 `OMNI_USER`/`OMNI_PASS`，或用项目的 URL 重定向 patch |
| 显存不足 / 启动前显存被占 | 僵尸 Isaac Sim 进程 | `nvidia-smi` 查 PID 后 `kill -9` |
| Arena 0.1.0 `pip install` 失败 | 0.1.0 **不提供 pip 包** | 只能走 Docker（见第 3 节） |
| GR00T 容器启动失败 | Blackwell GPU / DGX Spark 不受支持 | 改用 Base 容器 |

---

## 10. 参考来源

全部为 NVIDIA 官方文档与仓库（抓取于本指南编写时）：

1. [Isaac Lab — Installation using Isaac Lab Pip Packages (v2.3.2)](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/isaaclab_pip_installation.html)
2. [Isaac Lab — Installation using Isaac Sim Pre-built Binaries (v2.3.2)](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/binaries_installation.html)
3. [Isaac Sim 5.1.0 — Requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
4. [Isaac Sim — Download](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html)
5. [Isaac Lab-Arena — Installation (release/0.1.0)](https://isaac-sim.github.io/IsaacLab-Arena/release/0.1.0/pages/quickstart/installation.html)
6. [Isaac Lab-Arena — Docker Containers (release/0.1.0)](https://isaac-sim.github.io/IsaacLab-Arena/release/0.1.0/pages/quickstart/docker_containers.html)
7. [Isaac Lab-Arena — README（含版本兼容表）](https://github.com/isaac-sim/IsaacLab-Arena)
8. [Isaac Lab-Arena — Installing in Your Repository](https://isaac-sim.github.io/IsaacLab-Arena/main/pages/arena_in_your_repo/external_installation.html)

### 未能核实、需你确认的点

1. `unzip` 后 Isaac Sim 二进制的确切目录层级（官方仅假设 `${HOME}/isaacsim`）；
2. Isaac Lab 2.3 系列的 git tag 名（官方文档未直接给出）；
3. Isaac Sim 5.1 对应的 Docker 镜像 tag（`external_installation` 页用的是 6.0.0）；
4. `OMNI_KIT_ACCEPT_EULA=YES` / `ACCEPT_EULA=Y` 在 5.1 是否被识别；
5. Isaac Lab 2.3.2 与 2.3.0 该用哪个（安装页与 main 文档给了不同 patch 版本）；
6. Arena 0.1.0 究竟配 Isaac Sim 5.0.0 还是 5.1.0（安装页与 README 表冲突，见第 0 节）。
