"""A framework for analysis scale-out


"""
from .accumulator import (
    AccumulatorABC,
    accumulate,
    column_accumulator,
    defaultdict_accumulator,
    dict_accumulator,
    list_accumulator,
    set_accumulator,
    value_accumulator,
)
from .processor import ProcessorABC

__all__ = [
    "column_accumulator",
    "defaultdict_accumulator",
    "dict_accumulator",
    "list_accumulator",
    "set_accumulator",
    "value_accumulator",
    "accumulate",
    "AccumulatorABC",
    "ProcessorABC",
]
