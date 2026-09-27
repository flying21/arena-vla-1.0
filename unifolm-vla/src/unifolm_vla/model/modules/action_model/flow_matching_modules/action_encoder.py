# 麻雀虽小智能科技（武汉）有限公司
import torch
import torch.nn as nn


# =============================================================================
# 【中文说明】动作编码器与正弦位置编码
# =============================================================================
# 供 FlowmatchingActionHead 使用的两个基础模块：
#   SinusoidalPositionalEncoding —— 把标量时间步映射成正弦/余弦特征
#   swish / ActionEncoder        —— 把 (加噪动作, 时间步) 编码成 token 特征
#
# 【编码方式与经典 Transformer 的差别】
#   经典 Transformer 的正弦编码按**位置索引**取不同频率；
#   这里是把**标量数值**（归一化后的时间步或位置）本身映射成多频特征。
#   因此它既可用于动作序列的位置编码，也可用于 Flow Matching 的时间步编码。
#
# 【ActionEncoder 的融合方式】
#   a_emb = W1(actions)                 # 动作本身
#   tau_emb = pos_encoding(timesteps)   # 时间步
#   x = swish(W2([a_emb; tau_emb]))     # 拼接后降维 + 激活
#   out = W3(x)
#   即"早期拼接"（early fusion），让时间条件在进入 DiT 前就与动作混合。
# =============================================================================

def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


