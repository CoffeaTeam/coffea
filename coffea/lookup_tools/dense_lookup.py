from coffea.lookup_tools.lookup_base import lookup_base

import awkward
import numpy
from copy import deepcopy


class dense_lookup(lookup_base):
    def __init__(self, values, dims, feval_dim=None):
        super(dense_lookup, self).__init__()
        self._dimension = 0
        whattype = type(dims)
        if whattype == numpy.ndarray:
            self._dimension = 1
        else:
            self._dimension = len(dims)
        if self._dimension == 0:
            raise Exception("Could not define dimension for {}".format(whattype))
        self._axes = deepcopy(dims)
        self._feval_dim = None
        vals_are_strings = (
            "string" in values.dtype.name
            or "str" in values.dtype.name
            or "unicode" in values.dtype.name
            or "bytes" in values.dtype.name
        )  # ....
        if not isinstance(values, numpy.ndarray):
            raise TypeError("values is not a numpy array, but %r" % type(values))
        if vals_are_strings:
            raise Exception("dense_lookup cannot handle string values!")
        self._values = deepcopy(values)

    def _evaluate(self, *args):
        indices = []
        if self._dimension == 1:
            indices.append(
                numpy.clip(
                    numpy.searchsorted(self._axes, args[0], side="right") - 1,
                    0,
                    self._values.shape[0] - 1,
                )
            )
        else:
            for dim in range(self._dimension):
                indices.append(
                    numpy.clip(
                        numpy.searchsorted(self._axes[dim], args[dim], side="right")
                        - 1,
                        0,
                        self._values.shape[dim] - 1,
                    )
                )
        return self._values[tuple(indices)]

    def __repr__(self):
        myrepr = "{} dimensional histogram with axes:\n".format(self._dimension)
        temp = ""
        if self._dimension == 1:
            temp = "\t1: {}\n".format(self._axes)
        else:
            temp = "\t1: {}\n".format(self._axes[0])
        for idim in range(1, self._dimension):
            temp += "\t{}: {}\n".format(idim + 1, self._axes[idim])
        myrepr += temp
        return myrepr
