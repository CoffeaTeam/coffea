"""A framework for analysis scale-out


"""
from .accumulator import AccumulatorABC, accumulate, dict_accumulator, value_accumulator
from .processor import ProcessorABC

__all__ = [
    "dict_accumulator",
    "value_accumulator",
    "accumulate",
    "AccumulatorABC",
    "ProcessorABC",
]
