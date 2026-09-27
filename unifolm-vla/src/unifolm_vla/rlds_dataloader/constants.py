# 麻雀虽小智能科技（武汉）有限公司
"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# =============================================================================
# 【中文说明】训练/推理的维度与归一化常量 —— 全局唯一权威口径
# =============================================================================
# 本文件是**维度与归一化方式的真正来源**。ActionChunk 的动作维度、动作块长度、
# proprio 维度全部由这里决定；YAML 里的同名键只是记录，不起作用。
#
# 【平台自动探测机制】
#   detect_robot_platform() 把整个命令行拼成一个字符串，按关键词匹配平台：
#       "libero" -> LIBERO_CONSTANTS
#       "aloha"  -> ALOHA_CONSTANTS
#       "bridge" -> BRIDGE_CONSTANTS
#       "fractal"-> FRACTAL_CONSTANTS
#       "ee_6d"  -> G1_EE_6D_CONSTANTS
#       "joint"  -> G1_CONSTANTS
#       "stack_block" -> G1_STACK_BLOCK_CONSTANTS
#   无任何匹配时**默认 G1_EE_6D**（23 维动作 / 23 维 proprio / chunk=25）。
#
# 【★ 使用陷阱】
#   启动脚本里 data_mix 的命名会直接影响这里选哪套常量。例如训练 LIBERO 时
#   若命令行里不出现 "libero" 字样，就会错误回退到 G1_EE_6D（23 维），
#   导致维度与数据不匹配。训练 LIBERO 请确保参数中含 "libero"。
#
# 【各平台常量速查】
#   LIBERO   : chunk  8, action  7, proprio  8, BOUNDS_Q99
#   ALOHA    : chunk 25, action 14, proprio 14, BOUNDS
#   BRIDGE   : chunk  5, action  7, proprio  7, BOUNDS_Q99
#   FRACTAL  : chunk  5, action  7, proprio  8, BOUNDS_Q99
#   G1       : chunk 25, action 16, proprio 16, BOUNDS      （关节空间）
#   G1_EE_6D : chunk 25, action 23, proprio 23, BOUNDS_Q99  （末端位姿，Arena 用这个）
#   G1_STACK_BLOCK : chunk 25, action 23, proprio 23, BOUNDS_Q99
#
# 【另外两个常量】
#   ACTION_TOKEN_BEGIN_IDX=31743 与 ACTION_TOKEN_IDX=32001 属于"动作离散化 token"
#   方案（见 action_tokenizer.py）。本项目实际走连续动作的 Flow Matching，
#   因此这两个常量在当前链路上未被使用。
# =============================================================================

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'

# lisa method
ACTION_TOKEN_IDX = 32001

# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

FRACTAL_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

G1_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 16,
    "PROPRIO_DIM": 16,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

G1_EE_6D_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 23,
    "PROPRIO_DIM": 23,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

G1_STACK_BLOCK_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 23,
    "PROPRIO_DIM": 23,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()
    print(cmd_args)
    if "libero" in cmd_args:
        return "LIBERO"
    elif "aloha" in cmd_args:
        return "ALOHA"
    elif "bridge" in cmd_args:
        return "BRIDGE"
    elif "fractal" in cmd_args:
        return "FRACTAL"
    elif "ee_6d" in cmd_args:
        return "G1_EE_6D"
    elif "joint" in cmd_args:
        return "G1"
    elif "stack_block" in cmd_args:
        return "G1_STACK_BLOCK"
    else:
        return "G1_EE_6D"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "LIBERO":
    constants = LIBERO_CONSTANTS
elif ROBOT_PLATFORM == "ALOHA":
    constants = ALOHA_CONSTANTS
elif ROBOT_PLATFORM == "BRIDGE":
    constants = BRIDGE_CONSTANTS
elif ROBOT_PLATFORM == "FRACTAL":
    constants = FRACTAL_CONSTANTS
elif ROBOT_PLATFORM == "G1_EE_6D":
    constants = G1_EE_6D_CONSTANTS
elif ROBOT_PLATFORM == "G1":
    constants = G1_CONSTANTS
elif ROBOT_PLATFORM == "G1_STACK_BLOCK":
    constants = G1_STACK_BLOCK_CONSTANTS


# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f" in constants.py NUM_ACTIONS_CHUNK = {NUM_ACTIONS_CHUNK}")
print(f"  ACTION_DIM = {ACTION_DIM}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")
print("If needed, manually set the correct constants in `training/vla/constants.py`!")
