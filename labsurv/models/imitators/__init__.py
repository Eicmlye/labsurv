from .base_imitator import BaseImitator

# isort: off
# base class must be imported before children
from .airl import AIRL
from .gail import GAIL

# isort: on


__all__ = [
    "BaseImitator",
    "GAIL",
    "AIRL",
]
