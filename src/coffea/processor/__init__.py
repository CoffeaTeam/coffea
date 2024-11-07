"""A framework for analysis scale-out


"""

from .decorator import mapfilter
from .processor import ProcessorABC

__all__ = [
    "ProcessorABC",
    "mapfilter",
]
