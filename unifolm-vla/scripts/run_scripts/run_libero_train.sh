#!/usr/bin/env bash
# 麻雀虽小智能科技（武汉）有限公司
# =============================================================================
# UnifoLM-VLA 微调训练启动脚本（LIBERO 仿真基准）
# =============================================================================
#
# 与 run_unifolm_vla_train.sh 的差别（LIBERO 专用）：
#   window_size=2                  —— LIBERO 使用两帧历史观测
#   per_device_batch_size=16       —— 比 G1 数据用更大的批（数据更小）
#   data_mix=libero_*_no_noops     —— LIBERO 各任务套件
#
# 注意：constants.py 通过命令行参数探测平台，参数里含 "libero" 时会选中
#   LIBERO_CONSTANTS（NUM_ACTIONS_CHUNK=8, ACTION_DIM=7, PROPRIO_DIM=8,
#   BOUNDS_Q99），因此**必须**让 data_mix 里出现 "libero" 字样，
#   否则会错误地回退到 G1_EE_6D（23 维），导致维度不匹配。
# =============================================================================

# ---------------------------------------------------------------------------
# NCCL 通信配置（多机多卡 InfiniBand 环境相关）
# ---------------------------------------------------------------------------
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------
Framework_name=unifolm_vla
base_vlm=/path/to/your/Unifolm-VLM-0     # ★ 需修改：VLM 主干路径
model_type=qwen2_5_vl
freeze_module_list=''                    # 空 = 全量微调
window_size=2                            # LIBERO：使用两帧历史观测

# ---------------------------------------------------------------------------
# 数据集配置
# ---------------------------------------------------------------------------
oxe_data_root=/path/to/your/data         # ★ 需修改
data_mix=your_data_mix                   # ★ 需修改：libero_4_task_no_noops / libero_90_no_noops

# ---------------------------------------------------------------------------
# 产物目录
# ---------------------------------------------------------------------------
run_root_dir=/path/to/your/run_root_dir  # ★ 需修改
run_id=your_run_id                       # ★ 需修改

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/    # 备份启动脚本，便于复现

# ---------------------------------------------------------------------------
# 启动训练
# ---------------------------------------------------------------------------
accelerate launch \
  --config_file src/unifolm_vla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  src/unifolm_vla/training/train_unifolm_vla.py \
  --config_yaml ./src/unifolm_vla/config/training/unifolm_vla_train.yaml \
  --framework.framework_py ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.model_type ${model_type} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.window_size ${window_size} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 150000 \
  --trainer.shuffle_buffer_size 10000 \
  --trainer.save_interval 10000 \
  --trainer.use_wrist_image True \
  --trainer.use_proprio True \
  --trainer.logging_frequency 500 \
  --trainer.eval_interval 500 \
  --trainer.learning_rate.base 4e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project vla_jiang \
  --wandb_entity zbdz
