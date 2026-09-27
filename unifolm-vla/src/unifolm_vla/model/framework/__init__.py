# 麻雀虽小智能科技（武汉）有限公司
"""
Framework factory utilities.
Selects and instantiates a registered framework implementation based on config.
"""

# 【中文说明】框架工厂：按配置里的 framework.name 选择并实例化具体 VLA 实现。
#   通过 FRAMEWORK_REGISTRY（见 model/tools.py）查表，因此新增模型实现只需
#   用 @FRAMEWORK_REGISTRY.register("<名字>") 注册，无需改动本文件。

def build_framework(cfg):
    """
    Build a framework model from config.

    Args:
        cfg: Config object (OmegaConf / namespace) containing:
             cfg.framework.framework_py: Identifier string (e.g. "unifolm_vla")

    Returns:
        nn.Module: Instantiated framework model.

    Raises:
        NotImplementedError: If the specified framework id is unsupported.
    """
    if cfg.framework.framework_py == "unifolm_vla":
        from unifolm_vla.model.framework.unifolm_vla import Unifolm_VLA
        return Unifolm_VLA(cfg)
    
    raise NotImplementedError(f"Framework {cfg.framework.framework_py} is not implemented.")

