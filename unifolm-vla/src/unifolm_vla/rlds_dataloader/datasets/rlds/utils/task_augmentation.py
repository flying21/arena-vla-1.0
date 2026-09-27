# 麻雀虽小智能科技（武汉）有限公司
"""
task_augmentation.py

Contains basic logic for randomly zeroing out keys in the task specification.
"""

from typing import Dict

import tensorflow as tf

from unifolm_vla.rlds_dataloader.datasets.rlds.utils.data_utils import to_padding


# =============================================================================
# 【中文说明】任务描述增强（语言指令改写）
# =============================================================================
# 对训练样本的语言指令做改写增强，目的是让策略对**措辞变化**鲁棒，
# 而不是死记某一句话。典型手段：替换动词、增删修饰词、同义改写等。
#
# 【为什么重要】
#   推理时用户给的自然语言不会与训练标注逐字相同。若不做语言增强，
#   模型容易过拟合到具体句式，换一种说法就失效。
# =============================================================================

def delete_task_conditioning(traj: Dict, keep_image_prob: float) -> Dict:
    """
    Randomly drops out either the goal images or the language instruction. Only does something if both of
    these are present.

    Args:
        traj: A dictionary containing trajectory data. Should have a "task" key.
        keep_image_prob: The probability of keeping the goal images. The probability of keeping the language
            instruction is 1 - keep_image_prob.
    """
    if "language_instruction" not in traj["task"]:
        return traj

    image_keys = {key for key in traj["task"].keys() if key.startswith("image_") or key.startswith("depth_")}
    if not image_keys:
        return traj

    traj_len = tf.shape(traj["action"])[0]
    should_keep_images = tf.random.uniform([traj_len]) < keep_image_prob
    should_keep_images |= ~traj["task"]["pad_mask_dict"]["language_instruction"]

    for key in image_keys | {"language_instruction"}:
        should_keep = should_keep_images if key in image_keys else ~should_keep_images
        # pad out the key
        traj["task"][key] = tf.where(
            should_keep,
            traj["task"][key],
            to_padding(traj["task"][key]),
        )
        # zero out the pad mask dict for the key
        traj["task"]["pad_mask_dict"][key] = tf.where(
            should_keep,
            traj["task"]["pad_mask_dict"][key],
            tf.zeros_like(traj["task"]["pad_mask_dict"][key]),
        )

    # when no goal images are present, the goal timestep becomes the final timestep
    traj["task"]["timestep"] = tf.where(
        should_keep_images,
        traj["task"]["timestep"],
        traj_len - 1,
    )

    return traj
