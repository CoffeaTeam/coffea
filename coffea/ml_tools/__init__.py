"""Tools to interface with various ML inference services

Providing the interfaces to the run ML inference such that user can simply
handle data mangling in awkward/numpy formats. Specifics of passing numpy arrays
conversion and the handling of dask are mostly abstract away.
"""
from .triton_wrapper import triton_wrapper

__all__ = [
    "triton_wrapper",
    "LumiList",
    "LumiMask",
]
