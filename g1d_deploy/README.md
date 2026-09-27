# g1d_deploy —— 宇树 G1-D 真机抓取部署

本目录把工作区已有的三块能力（**ARENA VLA 框架** + **unifolm-vla 策略** +
**unitree_sdk2_python**）串成一条可上 G1-D 实体的抓取链路，且**不依赖 Isaac Lab**
（真机不需要仿真器）。

```
                    ┌──────────────────────────┐
                    │  VLA 策略服务器 (GPU)      │
                    │  POST /act  ← JSON/HTTP   │
                    └────────────▲─────────────┘
                                 │ action chunk
      observation (相机+关节)     │
   ┌─────────────────────────────┴─────────────────────────────┐
   │                      G1RealRobot                          │
   │  camera.py ─┬─▶ 观测    action_mapper.py ─▶ 关节目标        │
   │  lowstate ──┘             g1_lowlevel.py (500Hz DDS) ─▶ 机器人 │
   └────────────────────────────────────────────────────────────┘
```

## 目录内容

| 文件 | 职责 |
|------|------|
| `joint_map.py` | 关节槽位 / 动作空间 / 增益 / 限位的**唯一权威常量表** |
| `g1_lowlevel.py` | 500Hz 低层 DDS 关节控制（释放高层模式 + 平滑 + 看门狗 + 急停） |
| `action_mapper.py` | 动作 → 关节目标（16 维关节空间默认；23 维 EE 空间需注入 IK） |
| `camera.py` | 相机抽象（Mock / Webcam，可插拔机器人专属相机） |
| `g1_robot.py` | 实现 ARENA `RobotInterface` 的真机适配器 |
| `smoke_test.py` | **不接 VLA** 的低层链路冒烟测试（上电必跑） |
| `run_grasp.py` | 真机 VLA 闭环抓取入口 |
| `README.md` | 本文件 |

## 前置条件

- 机器人上位机（Linux，需 `timerfd`），已安装 `unitree_sdk2_python`（本工作区已带）与
  `cyclonedds`；
- 本机与机器人处于同一 DDS 域（默认 0）、网线直连且网卡名已知（如 `eth0`）；
- GPU 服务器已按 `unifolm-vla/deployment/model_server/run_real_eval_server.py` 启动
  VLA 服务，`curl http://<gpu>:8777/health` 返回 `200`；
- 已获得 checkpoint 的 `dataset_statistics.json`（动作/状态归一化用）。

## 快速开始

### 第 0 步：先验证低层链路（务必先做，不要直接上策略）

```bash
# 无硬件：仅验证控制律/映射逻辑
python -m g1d_deploy.smoke_test --dry-run --move

# 真机：保持姿态 2s，再做小幅摆臂测试（观察方向是否正确、是否稳定）
python -m g1d_deploy.smoke_test --interface eth0 --move
```

> 看到双臂关节角随正弦缓慢变化、且机器人稳定，才能进入下一步。

### 第 1 步：闭环抓取

```bash
python -m g1d_deploy.run_grasp \
    --interface eth0 \
    --server-url http://<gpu>:8777/act \
    --instruction "pick up the brown box and place it in the bin" \
    --norm-stats /path/to/checkpoint/dataset_statistics.json \
    --unnorm-key new_embodiment \
    --action-mode joint \
    --norm-type bounds \
    --max-steps 200
```

## 动作空间（与本工作区训练口径一致）

| 模式 | 维度 | 含义 | 需要 IK？ |
|------|------|------|-----------|
| `joint` | 16 | `[左臂7, 右臂7, 左夹爪, 右夹爪]` | 否（**推荐先跑这个**） |
| `ee` | 23 | 左/右 EEF 位置+四元数 + 导航/底座/躯干（见 DOCUMENTATION §12.2） | 是（需注入 `G1IK`） |

- `joint` 模式直接映射电机槽位，无需 IK，是真机首次上电最稳妥的路径；但要求 VLA
  checkpoint 是在**关节空间**训练的（启动服务端时命令行含 `joint`，见
  `unifolm_vla/rlds_dataloader/constants.py`）。
- `ee` 模式需要逆运动学。仿真里用的是 PINK IK，真机 SDK **不提供** IK；请实现
  `action_mapper.G1IK`（例如用 URDF + `ikpy`）后 `build_action_mapper("ee", ik=...)`。

## 标定清单（代码中用 `FIXME(标定)` 标出，必须按实际硬件核对）

1. **夹爪槽位与方向**：`joint_map.GRIPPER_SLOTS` 默认 `{left:29, right:30}` 是占位；
   `GRIPPER_POS_RANGE` 的"张开/闭合"方向必须按实际 G1-D 手部标定。
2. **关节限位**：`joint_map.ARM_JOINT_LIMITS` 是保守占位值，请以
   `g1_29dof_with_hand_rev_1_0.urdf` 覆盖。
3. **增益 Kp/Kd**：`joint_map.DEFAULT_KP/KD` 来自官方演示（非最优），按负载整定。
4. **proprio 顺序**：`g1_robot.build_state()` 的 16 维顺序必须与 checkpoint 的
   `dataset_statistics.json` 一致，否则动作会整体偏移（最难排查的静默错误）。
5. **相机**：G1 头部相机通常不是 SDK 内建服务，按实际接入方式实现
   `camera.CameraSource`（USB/RTSP/ROS 均可）。
6. **EE 路径 proprio**：`g1_robot.build_state()` 的 23 维是占位，需按 EE checkpoint 重排。

## 安全须知（重要）

- 低层控制**没有内置安全兜底**：`q/kp/kd` 写错、CRC 漏算都会导致抽动或摔倒。
- 首次上电建议**吊装或有人扶持**；随时准备按遥控器急停 / 切回阻尼 / 断电。
- 动作幅度从小开始：先用 `smoke_test.py` 验证方向，再逐步放大。
- 任何异常先跑 `robot.stop()`（Ctrl+C 会触发）。
