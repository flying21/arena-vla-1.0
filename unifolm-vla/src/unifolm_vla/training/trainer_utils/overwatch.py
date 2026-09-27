# 麻雀虽小智能科技（武汉）有限公司
"""
overwatch.py
# Original file from OpenVLA project (Prismatic), licensed under MIT License.
# See https://github.com/openvla/openvla for full license text and contributors.
# Modified by @JinhuiYE, [2025]
Utility class for creating a centralized/standardized logger (built on Rich) and accelerate handler.
"""

import logging
import logging.config
import os
from contextlib import nullcontext
from logging import LoggerAdapter
from typing import Any, Callable, ClassVar, Dict, MutableMapping, Tuple, Union

# Overwatch Default Format String
# =============================================================================
# 【中文说明】日志封装（Overwatch）
# =============================================================================
# 官方 OpenVLA/Open-X 体系里的日志工具：把标准 logging 包一层，
# 通过环境变量控制日志等级与是否启用富文本/进度条。
#
# 本项目的使用方式：
#   from unifolm_vla.training.trainer_utils import initialize_overwatch
#   logger = initialize_overwatch(__name__)
# 之后用 logger.info(...) / logger.warning(...) 输出。
#
# 【为什么要有这一层】分布式训练下需要：
#   - 只在主进程打印（避免 N 份重复日志）；
#   - 统一抑制第三方库的冗余输出；
#   - 支持 rank 前缀。
# =============================================================================

RICH_FORMATTER, DATEFMT = "| >> %(message)s", "%m/%d [%H:%M:%S]"

# Set Logging Configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {"simple-console": {"format": RICH_FORMATTER, "datefmt": DATEFMT}},
    "handlers": {
        "console": {
            "class": "rich.logging.RichHandler",
            "formatter": "simple-console",
            "markup": True,
            "rich_tracebacks": True,
            "show_level": True,
            "show_path": True,
            "show_time": True,
        }
    },
    "root": {"level": "INFO", "handlers": ["console"]},
}
logging.config.dictConfig(LOG_CONFIG)


# === Custom Contextual Logging Logic ===
class ContextAdapter(LoggerAdapter):
    CTX_PREFIXES: ClassVar[Dict[int, str]] = {**{0: "[*] "}, **{idx: "|=> ".rjust(4 + (idx * 4)) for idx in [1, 2, 3]}}

    def process(self, msg: str, kwargs: MutableMapping[str, Any]) -> Tuple[str, MutableMapping[str, Any]]:
        ctx_level = kwargs.pop("ctx_level", 0)
        return f"{self.CTX_PREFIXES[ctx_level]}{msg}", kwargs


class DistributedOverwatch:
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that wraps logging & `accelerate.PartialState`."""
        from accelerate import PartialState

        # Note that PartialState is always safe to initialize regardless of `accelerate launch` or `torchrun`
        #   =>> However, might be worth actually figuring out if we need the `accelerate` dependency at all!
        self.logger, self.distributed_state = ContextAdapter(logging.getLogger(name), extra={}), PartialState()

        # Logger Delegation
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> only Log `INFO` on Main Process, `ERROR` on others!
        self.logger.setLevel(logging.INFO if self.distributed_state.is_main_process else logging.ERROR)

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        return self.distributed_state.on_main_process

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        return self.distributed_state.on_local_main_process

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        return self.distributed_state.main_process_first

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        return self.distributed_state.local_main_process_first

    def is_rank_zero(self) -> bool:
        return self.distributed_state.is_main_process

    def rank(self) -> int:
        return self.distributed_state.process_index

    def local_rank(self) -> int:
        return self.distributed_state.local_process_index

    def world_size(self) -> int:
        return self.distributed_state.num_processes


class PureOverwatch:
    def __init__(self, name: str) -> None:
        """Initializer for an Overwatch object that just wraps logging."""
        self.logger = ContextAdapter(logging.getLogger(name), extra={})

        # Logger Delegation
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        # Logging Defaults =>> INFO
        self.logger.setLevel(logging.INFO)

    @staticmethod
    def get_identity_ctx() -> Callable[..., Any]:
        def identity(fn: Callable[..., Any]) -> Callable[..., Any]:
            return fn

        return identity

    @property
    def rank_zero_only(self) -> Callable[..., Any]:
        return self.get_identity_ctx()

    @property
    def local_zero_only(self) -> Callable[..., Any]:
        return self.get_identity_ctx()

    @property
    def rank_zero_first(self) -> Callable[..., Any]:
        return nullcontext

    @property
    def local_zero_first(self) -> Callable[..., Any]:
        return nullcontext

    @staticmethod
    def is_rank_zero() -> bool:
        return True

    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def world_size() -> int:
        return 1


def initialize_overwatch(name: str) -> Union[DistributedOverwatch, PureOverwatch]:
    return DistributedOverwatch(name) if int(os.environ.get("WORLD_SIZE", -1)) != -1 else PureOverwatch(name)
