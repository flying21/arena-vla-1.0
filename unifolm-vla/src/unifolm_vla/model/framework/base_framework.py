# 麻雀虽小智能科技（武汉）有限公司
# =============================================================================
# 【中文说明】VLA 模型装配基类：checkpoint 加载与归一化统计量管理
# =============================================================================
# 本模块提供 baseframework，是 Unifolm_VLA 的父类。它只做两件事：
#   1. 从 checkpoint 恢复模型（from_pretrained）；
#   2. 管理数据集归一化统计量的选择与校验（get_action_stats）。
#
# 【checkpoint 目录契约 —— 硬性要求】
#   from_pretrained 收的是 .pt 路径，但真正需要的是它**上一级目录**里的另两个文件：
#
#     <run_dir>/
#       ├── config.yaml               # 模型结构配置（或 config.json）
#       ├── dataset_statistics.json   # 动作/状态的归一化统计量
#       └── checkpoints/
#             └── pytorch_model.pt    # 权重
#
#   ★ 最常见的失败原因就是"只拷了 .pt 文件"——缺 config 或统计量会直接报错。
#     读取逻辑见 share_tools.read_mode_config。
#
# 【为什么统计量必须和模型一起加载】
#   推理必须用训练时**同一套**统计量做反归一化，否则动作会整体偏移/缩放却
#   不报任何错，属于最难排查的静默错误。因此统计量被挂在模型实例上
#   （FrameworkModel.norm_stats），由推理侧按任务键取用。
#
# 【本文件已知的重复定义（不改动，仅为阅读提示）】
#   _check_unnorm_key 与 get_action_stats 各定义了两次，后者覆盖前者。
#   两版语义等价，第二版 get_action_stats 多一个 norm_stats 显式入参。
#   这是上游遗留，保留原样以免偏离 vendored 代码。
# =============================================================================

import torch.nn as nn
from typing import List

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from typing import List

from pathlib import Path
from typing import Dict, List

import numpy as np
from unifolm_vla.model.tools import auto_get_trainable_modules

from unifolm_vla.model.framework.share_tools import read_mode_config
from unifolm_vla.training.trainer_utils import initialize_overwatch
from unifolm_vla.model.framework.share_tools import dict_to_namespace
from unifolm_vla.model.framework.__init__ import build_framework

logger = initialize_overwatch(__name__)


class baseframework(nn.Module):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(
        self,
    ) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """
        super().__init__()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        vlm_pretrained_path: str = None,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict strictly (reports missing/unexpected keys)
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint)  # read config and norm_stats

        config = dict_to_namespace(model_config)
        model_config = config
        if vlm_pretrained_path is not None:
            model_config.framework.qwenvl.base_vlm = vlm_pretrained_path
        FrameworkModel = build_framework(cfg=model_config)
        FrameworkModel.norm_stats = norm_stats
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
        model_keys = set(FrameworkModel.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        try:
            FrameworkModel.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            common_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

            raise e

        FrameworkModel = FrameworkModel
        return FrameworkModel

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Infer or validate the dataset stats key used for un-normalization.

        Args:
            norm_stats: Dict[str, dict] mapping dataset key -> stats block.
            unnorm_key: Optional explicit dataset key.

        Returns:
            str: Resolved key.

        Raises:
            AssertionError: If multiple datasets present and key not provided,
                            or provided key not found.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @classmethod
    def get_action_stats(self, unnorm_key=None):
        """
        Retrieve raw action normalization statistics.

        Args:
            unnorm_key: Optional dataset stats key.

        Returns:
            dict: Stats structure (e.g. q01, q99, mask).
        """
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]


    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @classmethod
    def get_action_stats(self, unnorm_key=None, norm_stats=None):
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        if norm_stats ==None:
            norm_stats = self.norm_stats
        unnorm_key = self._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]
