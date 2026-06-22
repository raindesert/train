"""公共工具."""
from .device import get_device, get_dtype
from .logging import get_logger, setup_logging
from .seed import set_seed

__all__ = ["get_device", "get_dtype", "get_logger", "setup_logging", "set_seed"]
