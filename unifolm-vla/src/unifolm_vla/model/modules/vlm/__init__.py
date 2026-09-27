# 麻雀虽小智能科技（武汉）有限公司
# 【中文说明】VLM 子包入口：导出 get_vlm_model 工厂函数（当前实现为 Qwen2.5-VL）。

def get_vlm_model(config):
    
    vlm_name = config.framework.qwenvl.model_type
    if vlm_name == "qwen2_5_vl":
        from .QWen2_5 import _QWen_VL_Interface 
        return _QWen_VL_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")



