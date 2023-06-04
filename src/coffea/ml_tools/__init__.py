"""Tools to interface with various ML inference services

Providing the interfaces to the run ML inference such that user can simply
handle data mangling in awkward/numpy formats. Specifics of passing numpy arrays
conversion and the handling of dask are mostly abstract away.
"""
from .helper import numpy_call_wrapper

__all__ = [
    "numpy_call_wrapper",
    "torch_wrapper",
    "triton_wrapper",
    "xgboost_wrapper",
]
