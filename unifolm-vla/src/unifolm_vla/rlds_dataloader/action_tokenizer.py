# 麻雀虽小智能科技（武汉）有限公司
"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
"""

from typing import List, Union

import numpy as np


# =============================================================================
# 【中文说明】动作离散化分词器（当前主链路未使用）
# =============================================================================
# 把连续动作按**均匀分箱**离散化，再映射成词表末端的 token 字符串，
# 形如 "<action_128>"。这是"把动作当语言 token 预测"那一类 VLA 的做法
# （对应 constants.py 里的 ACTION_TOKEN_BEGIN_IDX / ACTION_TOKEN_IDX）。
#
# 【分箱与还原约定】
#   bins = linspace(min_action, max_action, n_bins)   # n_bins=256
#   bin_centers = 相邻 bins 的中点                    # 共 n_bins-1 个
#   __call__: np.digitize 得到 1..n_bins 的箱索引，拼成字符串
#   decode_token_ids_to_actions: 索引减 1 并裁剪到 [0, n_bins-2]，
#     再查 bin_centers 还原成连续值
#   ★ 用 digitize 而非查中心点，所以索引范围比中心点数量多 1，
#     代码里必须做"减 1 + 上限裁剪"，否则会越界。
#
# 【为什么本项目不用它】
#   UnifoLM-VLA 走的是连续动作的 **Flow Matching**（见 DiT_ActionHeader.py），
#   不需要把动作离散化成 token。本文件保留是为了兼容上游与方便对照。
# =============================================================================

class ActionTokenizer:
    def __init__(self, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """
        self.n_bins, self.min_action, self.max_action = bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0


    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins) # [8, 14]
        
        return f"<action_{discretized_action}>"
        
        
        # if len(discretized_action.shape) == 1:
        #     return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        # else:
        #     return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, discretized_actions: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins
    


