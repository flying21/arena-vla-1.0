#!/usr/bin/env bash
# 麻雀虽小智能科技（武汉）有限公司
# =============================================================================
# UnifoLM-VLA 在 LIBERO 仿真基准上的评测脚本
# =============================================================================
# 用 UnifoLM-VLA-Libero 权重在 LIBERO 四个任务套件上跑评测，
# 每个任务重复 num_trials_per_task 次，并录制视频。
#
# 【前置条件（缺一不可）】
#   1. 已安装 LIBERO 本体：
#        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
#        pip install -e LIBERO
#        pip install -r experiments/LIBERO/libero_requirements.txt
#   2. experiments/LIBERO/eval_libero.py 存在
#      ★ 注意：本工作区的 unifolm-vla 副本**已删除 experiments/ 目录**，
#        因此本脚本在当前工作区**无法直接运行**，需先从上游仓库取回该目录。
#
# 【维度一致性提醒】
#   constants.py 通过命令行参数探测平台。本脚本调用的是 eval_libero.py，
#   并由 unnorm_key 指定 LIBERO 统计量，但**本脚本未把 "libero" 字样传入
#   训练/常量探测用的命令行**——若在别的入口复用类似写法，务必确认
#   constants 选中了 LIBERO_CONSTANTS(7/8/chunk8) 而不是 G1_EE_6D(23/23/25)。
# =============================================================================

# ---------------------------------------------------------------------------
# LIBERO 环境路径（★ 需按实际安装位置修改）
# ---------------------------------------------------------------------------
export LIBERO_HOME=/jfs/jiang/code/unitree/LIBERO   # ★ 需修改：LIBERO 仓库路径
# export LIBERO_HOME=/path/to/your/LIBERO            # 可替换为本地路径
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero      # LIBERO 的配置目录

# 让 python 既能 import libero，也能 import 本项目（$(pwd) 即 unifolm-vla 根目录）
export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}

# ---------------------------------------------------------------------------
# 模型与 checkpoint（★ 需按实际路径修改）
# ---------------------------------------------------------------------------
# your_ckpt=/path/to/your/Unifolm-VLA-Libero/checkpoints/pytorch_model.pt
# vlm_pretrained_path=/path/to/your/Unifolm-VLM-Base
your_ckpt=/DATA/disk2/unitree_vla/unitreevla_libero_4_task_window_size_2/checkpoints/pytorch_model.pt
vlm_pretrained_path=/root/Unifolm-VLM-0

# 从 checkpoint 路径中截取目录名/步数名，仅用于组织输出目录
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $5}')
step_name=$(echo "$your_ckpt" | awk -F'/' '{print $6}')

# ---------------------------------------------------------------------------
# 评测配置
# ---------------------------------------------------------------------------
task_suite_name=libero_spatial   # 可选：libero_goal, libero_object, libero_10, libero_90
num_trials_per_task=50           # 每个任务重复次数（决定成功率统计的分辨率）
window_size=2                    # LIBERO 使用两帧历史观测（与训练脚本一致）
# 反归一化所用的数据集键，必须与 task_suite 对应
unnorm_key="libero_spatial_no_noops"  # 可选：libero_goal_no_noops, libero_object_no_noops, libero_10_no_noops, libero_90_no_noops

# 视频输出目录：results/<任务套件>/<checkpoint 目录>/<步数>/
video_out_path="results/${task_suite_name}/${folder_name}/${step_name}"

DEVICE=0   # 使用的 GPU 编号

# ---------------------------------------------------------------------------
# 启动评测
# ---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=${DEVICE} python ./experiments/LIBERO/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.vlm-pretrained-path ${vlm_pretrained_path} \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    --args.unnorm-key "$unnorm_key" \
    --args.window-size "$window_size"
