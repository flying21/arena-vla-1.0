# 麻雀虽小智能科技（武汉）有限公司
"""
traj_transforms.py

Contains trajectory transforms used in the orca data pipeline. Trajectory transforms operate on a dictionary
that represents a single trajectory, meaning each tensor has the same leading dimension (the trajectory length).
"""

import logging
from typing import Dict

import tensorflow as tf


# =============================================================================
# 【中文说明】轨迹级变换：动作分块（action chunking）是核心
# =============================================================================
# 【chunk_act_obs —— 本项目动作块的来源】
#   按官方文档：观测获得长度 window_size 的轴（过去 window_size-1 帧 + 当前），
#   动作获得长度 window_size + future_action_window_size 的轴
#   （过去 window_size-1 个 + 当前 + 未来 future_action_window_size 个）。
#
#   实现要点：
#   - chunk_indices 用广播构造每个时间点的滑窗索引；
#   - floored_* 把负索引夹到 0（轨迹开头用首帧补齐）；
#   - 末尾用 goal_timestep 夹住，超出轨迹末尾的动作变成"中性动作"。
#
#   ★ 上游在此处改过 goal timestep 的定义（注释里保留了原代码）：
#     原实现会读 traj["task"]["timestep"]，现改为 goal_timestep = traj_len - 1，
#     即把整条轨迹的终点当作目标时刻。这会影响末端动作的填充方式。
#
#   ★ absolute_action_mask 缺失时只 warning 不报错，默认把全部动作视为
#     "相对动作"（relative）：相对动作在填充时置零，绝对动作则重复当前值。
#     若你的数据集动作定义是绝对的却没提供该掩码，尾部填充语义就是错的。
#
# 【subsample / add_pad_mask_dict】
#   前者做轨迹下采样；后者为 observation/task 的每个字段生成 padding 掩码，
#   字符串字段按"长度是否为 0"判断，其余一律视为非 padding。
# =============================================================================

def chunk_act_obs(traj: Dict, window_size: int, future_action_window_size: int = 0) -> Dict:
    """
    Chunks actions and observations into the given window_size.

    "observation" keys are given a new axis (at index 1) of size `window_size` containing `window_size - 1`
    observations from the past and the current observation. "action" is given a new axis (at index 1) of size
    `window_size + future_action_window_size` containing `window_size - 1` actions from the past, the current
    action, and `future_action_window_size` actions from the future. "pad_mask" is added to "observation" and
    indicates whether an observation should be considered padding (i.e. if it had come from a timestep
    before the start of the trajectory).
    """
    traj_len = tf.shape(traj["action"])[0]
    action_dim = traj["action"].shape[-1]
    chunk_indices = tf.broadcast_to(tf.range(-window_size + 1, 1), [traj_len, window_size]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, window_size]
    )

    action_chunk_indices = tf.broadcast_to(
        tf.range(-window_size + 1, 1 + future_action_window_size),
        [traj_len, window_size + future_action_window_size],
    ) + tf.broadcast_to(
        tf.range(traj_len)[:, None],
        [traj_len, window_size + future_action_window_size],
    )
    floored_chunk_indices = tf.maximum(chunk_indices, 0)
    # chiayu editted. modify goal timestep definition
    ## There is something weird in the goal timestep definition.
   # if "timestep" in traj["task"]:
    #    goal_timestep = traj["task"]["timestep"]
   # else:
    goal_timestep = tf.fill([traj_len], traj_len - 1)

    floored_action_chunk_indices = tf.minimum(tf.maximum(action_chunk_indices, 0), goal_timestep[:, None])
    traj["observation"] = tf.nest.map_structure(lambda x: tf.gather(x, floored_chunk_indices), traj["observation"])
    traj["action"] = tf.gather(traj["action"], floored_action_chunk_indices)

    # indicates whether an entire observation is padding
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # if no absolute_action_mask was provided, assume all actions are relative
    if "absolute_action_mask" not in traj and future_action_window_size > 0:
        logging.warning(
            "future_action_window_size > 0 but no absolute_action_mask was provided. "
            "Assuming all actions are relative for the purpose of making neutral actions."
        )
    absolute_action_mask = traj.get("absolute_action_mask", tf.zeros([traj_len, action_dim], dtype=tf.bool))

    neutral_actions = tf.where(
        absolute_action_mask[:, None, :],
        traj["action"],  # absolute actions are repeated (already done during chunking)
        tf.zeros_like(traj["action"]),  # relative actions are zeroed
    )

    # actions past the goal timestep become neutral
    action_past_goal = action_chunk_indices > goal_timestep[:, None]
    traj["action"] = tf.where(action_past_goal[:, :, None], neutral_actions, traj["action"])
    #traj['check'] = goal_timestep
    return traj




def subsample(traj: Dict, subsample_length: int) -> Dict:
    """Subsamples trajectories to the given length."""
    traj_len = tf.shape(traj["action"])[0]
    if traj_len > subsample_length:
        indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
        traj = tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)

    return traj


def add_pad_mask_dict(traj: Dict) -> Dict:
    """
    Adds a dictionary indicating which elements of the observation/task should be treated as padding.
        =>> traj["observation"|"task"]["pad_mask_dict"] = {k: traj["observation"|"task"][k] is not padding}
    """
    traj_len = tf.shape(traj["action"])[0]

    for key in ["observation", "task"]:
        pad_mask_dict = {}
        for subkey in traj[key]:
            # Handles "language_instruction", "image_*", and "depth_*"
            if traj[key][subkey].dtype == tf.string:
                pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0

            # All other keys should not be treated as padding
            else:
                pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)

        traj[key]["pad_mask_dict"] = pad_mask_dict

    return traj
