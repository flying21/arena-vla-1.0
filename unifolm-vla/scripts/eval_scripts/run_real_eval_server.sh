#!/usr/bin/env bash
# 麻雀虽小智能科技（武汉）有限公司
# =============================================================================
# 启动 UnifoLM-VLA 真机推理服务器（FastAPI）
# =============================================================================
#
# 部署形态（与技术报告一致）：
#   GPU 机器：跑本脚本，模型推理在服务端执行
#   机器人端：跑 unitree_deploy 客户端，采集观测并 POST 到本服务器的 /act
#
# 两端通常不在同一台机器，标准做法是建立 SSH 隧道：
#   ssh user@remote_server_IP -CNg -L 8777:127.0.0.1:8777
#
# 端点：
#   POST /act     推理（请求体 {"observations": [...], "instruction": str}）
#   GET  /health  真实探活：模型就绪 -> 200，否则 503
#
# 必须先确认 checkpoint 目录结构（share_tools.read_mode_config 的硬性要求）：
#   models/UnifoLM-VLA-Base1/
#     ├── config.yaml              <-- 与 .pt 同级的上一级目录
#     ├── dataset_statistics.json  <-- 反归一化统计量，缺它必然失败
#     └── checkpoints/pytorch_model.pt
# =============================================================================

python deployment/model_server/run_real_eval_server.py \
    --ckpt_path models/UnifoLM-VLA-Base1/checkpoints/pytorch_model.pt \
    --port 8777 \
    --unnorm_key g1_stack_block \
    --vlm_pretrained_path models/UnifoLM-VLM-Base
