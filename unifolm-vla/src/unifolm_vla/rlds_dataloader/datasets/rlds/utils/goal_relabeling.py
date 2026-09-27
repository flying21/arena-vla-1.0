# 麻雀虽小智能科技（武汉）有限公司
"""
goal_relabeling.py

Contains simple goal relabeling logic for BC use-cases where rewards and next_observations are not required.
Each function should add entries to the "task" dict.
"""

from typing import Dict

import tensorflow as tf

from unifolm_vla.rlds_dataloader.datasets.rlds.utils.data_utils import tree_merge


# =============================================================================
# 【中文说明】目标重标注（goal relabeling）
# =============================================================================
# 训练时把轨迹中"更靠后的某一帧"当作目标，从而：
#   - 用同一段数据合成出多个"不同起点 -> 同一目标"的学习信号；
#   - 提高数据利用效率（HER 思想在模仿学习中的变体）。
#
# 本项目在 RLDSDataset 里传的策略是 "uniform"（在轨迹长度上均匀采目标时刻）。
#
# 【本项目实际未使用目标信息】
#   当前 VLA 走的是"观测 + 语言 -> 动作块"的直接映射，并不输入目标图像，
#   因此重标注主要用于兼容上游管线；动作分块本身已提供时间维度的监督。
# =============================================================================

def uniform(traj: Dict) -> Dict:
    """Relabels with a true uniform distribution over future states."""
    traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]

    # Select a random future index for each transition i in the range [i + 1, traj_len)
    rand = tf.random.uniform([traj_len])
    low = tf.cast(tf.range(traj_len) + 1, tf.float32)
    high = tf.cast(traj_len, tf.float32)
    goal_idxs = tf.cast(rand * (high - low) + low, tf.int32)

    # Sometimes there are floating-point errors that cause an out-of-bounds
    goal_idxs = tf.minimum(goal_idxs, traj_len - 1)

    # Adds keys to "task" mirroring "observation" keys (`tree_merge` to combine "pad_mask_dict" properly)
    goal = tf.nest.map_structure(lambda x: tf.gather(x, goal_idxs), traj["observation"])
    traj["task"] = tree_merge(traj["task"], goal)

    return traj
