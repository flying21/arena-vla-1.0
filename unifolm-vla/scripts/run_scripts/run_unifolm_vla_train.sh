#!/usr/bin/env bash
# 麻雀虽小智能科技（武汉）有限公司
# =============================================================================
# UnifoLM-VLA 微调训练启动脚本（宇树开源数据集 / 自定义 RLDS 数据）
# =============================================================================
#
# 作用：用 accelerate + DeepSpeed ZeRO-2 在 8 卡上微调 UnifoLM-VLA。
#
# 使用前必须修改的变量（下面标了 "★ 需修改"）：
#   base_vlm        —— 视觉语言主干 UnifoLM-VLM-Base 的本地路径
#   oxe_data_root   —— RLDS 数据集根目录
#   data_mix        —— 参与训练的数据集组合名或数据集名
#   run_root_dir    —— 训练产物（checkpoint / 日志）保存根目录
#   run_id          —— 本次运行的标识（决定输出子目录名）
#
# 数据准备链路（三步，详见 unifolm-vla/README_cn.md）：
#   LeRobot V2.1 格式
#     --prepare_data/convert_lerobot_to_hdf5.py-->  HDF5
#     --prepare_data/hdf5_to_rlds + tfds build-->  RLDS (1.0.0)
#
# 注意：维度常量不在本脚本、也不在 YAML 里，而在
#   src/unifolm_vla/rlds_dataloader/constants.py
# 该文件会**根据命令行参数自动探测平台**（detect_robot_platform），
# 无匹配时回退 G1_EE_6D（ACTION_DIM=23, PROPRIO_DIM=23, chunk=25）。
# =============================================================================

# ---------------------------------------------------------------------------
# NCCL 通信配置（多机多卡 InfiniBand 环境相关）
# ---------------------------------------------------------------------------
export NCCL_SOCKET_IFNAME=bond0          # 绑定的网卡（按实际集群修改）
export NCCL_IB_HCA=mlx5_2,mlx5_3         # 使用的 IB 网卡
export NCCL_BLOCKING_WAIT=1              # 阻塞式等待，便于定位卡死
export NCCL_ASYNC_ERROR_HANDLING=1       # 异步错误处理
export NCCL_TIMEOUT=1000                 # 通信超时（秒），大模型训练需要放宽

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------
Framework_name=unifolm_vla
base_vlm=/path/to/your/Unifolm-VLM-0     # ★ 需修改：VLM 主干路径
model_type=qwen2_5_vl
freeze_module_list=''                    # 空 = 不冻结任何模块（全量微调）
window_size=1                            # 每样本送几帧历史观测（LIBERO 用 2）

# ---------------------------------------------------------------------------
# 数据集配置
# ---------------------------------------------------------------------------
oxe_data_root=/path/to/your/data         # ★ 需修改：RLDS 数据根目录
data_mix=your_data_mix                   # ★ 需修改：如 Unitree_all_task / g1_stack_block

# ---------------------------------------------------------------------------
# 产物目录：输出到 ${run_root_dir}/${run_id}/
# ---------------------------------------------------------------------------
run_root_dir=/path/to/your/run_root_dir  # ★ 需修改
run_id=your_run_id                       # ★ 需修改

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/    # 备份本脚本，保证训练结果可追溯到当时的启动参数

# ---------------------------------------------------------------------------
# 启动训练
# ---------------------------------------------------------------------------
# accelerate launch 负责按 num_processes 拉起分布式进程；
# 每个 "--a.b value" 参数会经 normalize_dotlist_args 转成 OmegaConf 覆盖项，
# 合并到 --config_yaml 指向的基础配置之上（命令行优先级更高）。
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
  --datasets.vla_data.per_device_batch_size 6 \
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
