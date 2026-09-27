# 麻雀虽小智能科技（武汉）有限公司
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from unifolm_vla.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

from unifolm_vla.model.framework.base_framework import baseframework
from unifolm_vla.model.modules.vlm import get_vlm_model
from unifolm_vla.model.modules.action_model.DiT_ActionHeader import get_action_model, FlowmatchingActionHead
from unifolm_vla.model.tools import FRAMEWORK_REGISTRY

# =============================================================================
# 【中文说明】UnifoLM-VLA 顶层模型：Qwen2.5-VL 主干 + Flow Matching 动作头
# =============================================================================
# 本类把两个部件拼成一个 VLA：
#   qwen_vl_interface  —— 视觉语言主干（Qwen2.5-VL），把「图像 + 文本」编码成隐状态
#   action_model       —— FlowmatchingActionHead（含 DiT），把隐状态解码成动作块
#
# 【关键设计：动作头不是分类头】
#   它不做 token 预测，而是做**连续动作的流匹配（Flow Matching）**：
#     训练：从 Beta 分布采样时间步 t，构造加噪动作 (1-t)*noise + t*actions，
#           让网络预测速度场 velocity = actions - noise，损失为 MSE。
#     推理：从纯噪声出发，用 Euler 积分迭代 num_inference_timesteps(默认4) 步去噪。
#   相比扩散模型，流匹配步数更少、推理更快，是当前 VLA 的主流选择之一。
#
# 【两个容易踩的点】
#   1. cross_attention_dim 必须等于 VLM 的 hidden_size。本文件在 __init__ 里
#      **运行时覆盖**配置值：self.config.framework.action_model.diffusion_model_cfg
#      .cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
#      —— 所以 YAML 里那个 2048 只是记录，真实值来自模型。
#   2. 动作维度 / 动作块长度 / proprio 维度**不取自配置**，而是从
#      rlds_dataloader/constants.py 读取全局常量 ACTION_DIM / PROPRIO_DIM /
#      NUM_ACTIONS_CHUNK（见 DiT_ActionHeader.py）。constants.py 会按命令行参数
#      自动探测平台，无匹配时回退 G1_EE_6D（23/23/25）。
#
# 【repeated_diffusion_steps 的陷阱】
#   本文件读的是 self.config.**trainer**.repeated_diffusion_steps，
#   而 YAML 里 framework.action_model.repeated_diffusion_steps 那个键**不生效**。
# =============================================================================

@FRAMEWORK_REGISTRY.register("unifolm_vla")
class Unifolm_VLA(baseframework):
    """
    Multimodal vision-language-action model.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size
        self.processor = self.qwen_vl_interface.processor
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  
    
    def forward(
        self,
        qwen_inputs: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        actions = qwen_inputs["action"].to(torch.bfloat16)
        state = qwen_inputs["state"].to(torch.bfloat16) 
        state = state.unsqueeze(1) if state.dim() == 2 else state

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs["pixel_values"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                output_hidden_states = True,       
                return_dict=True,  
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            
        with torch.autocast("cuda", dtype=torch.float32):
            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            if state is not None:
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)
            action_loss = self.action_model(
                last_hidden_repeated, 
                actions_target_repeated, 
                state_repeated,
            )
            
        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        qwen_inputs,
        **kwargs: str,
    ) -> np.ndarray:

        state = qwen_inputs["state"]
        state = state.unsqueeze(1) if state.dim() == 2 else state

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                input_ids=qwen_inputs["input_ids"],
                attention_mask=qwen_inputs["attention_mask"],
                pixel_values=qwen_inputs["pixel_values"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                output_hidden_states = True,       
                return_dict=True,  
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            
        state = state.to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden, 
                state,
            )
        normalized_actions = pred_actions.detach().cpu().numpy()
        
        return {"normalized_actions": normalized_actions}

