"""A framework for analysis scale-out


"""
from .accumulator import AccumulatorABC, dict_accumulator
from .processor import ProcessorABC

__all__ = [
    "dict_accumulator",
    "AccumulatorABC",
    "ProcessorABC",
]
