"""Awkward 0.x NanoEvents and helpers

See `coffea.nanoevents` for an awkward1-based implementation

This package provides NanoEvents, a utility to wrap the CMS NanoAOD (or similar)
flat nTuple structure into a single awkward array with appropriate object methods
(such as Lorentz vector methods), cross references, and pre-built nested objects,
all lazily accessed from the source ROOT TTree via uproot.

NanoEvents is in a **experimental** stage at this point. Certain functionality may be
fragile, and some functionality will not be available until it is ported to awkward-array version 1.
"""
from .nanoevents import NanoCollection, NanoEvents

__all__ = [
    'NanoCollection',
    'NanoEvents',
]
