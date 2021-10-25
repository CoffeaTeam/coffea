from coffea.lookup_tools.lookup_base import lookup_base

import numpy
import correctionlib
from copy import deepcopy


class correctionlib_shim(lookup_base):
    def __init__(self, corr):
        self._corr = corr

    def _evaluate(self, *args):
        return self._corr.evaluate(*args)

    def __repr__(self):
        signature = ",".join(
            inp.name if len(inp.name) > 0 else f"input{i}"
            for i, inp in enumerate(self._corr._base.inputs)
        )
        return f"correctionlib Correction: {self._corr.name}({signature})"


class correctionlib_wrapper(lookup_base):
    def __init__(self, payload):
        super(correctionlib_wrapper, self).__init__()
        self._cset = payload.to_evaluator()

    def __getitem__(self, key):
        return correctionlib_shim(self._cset[key])

    def keys(self):
        return self._cset.keys()

    def _evaluate(self, *args):
        raise RuntimeError(
            "correctionlib_wrapper is not callable, choose a correction via wrapper['corr-name']"
        )

    def __repr__(self):
        myrepr = f"correctionlib CorrectionSet with {len(self.keys())} corrections:\n"
        for name in self._cset:
            myrepr += f"\t{name} with {len(self._cset[name]._base.inputs)}\n"
        return myrepr
