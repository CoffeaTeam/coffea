from coffea.lookup_tools.lookup_base import lookup_base


class correctionlib_wrapper(lookup_base):
    def __init__(self, payload):
        super(correctionlib_wrapper, self).__init__()
        self._corr = payload

    def _evaluate(self, *args):
        return self._corr.evaluate(*args)

    def __repr__(self):
        signature = ",".join(
            inp.name if len(inp.name) > 0 else f"input{i}"
            for i, inp in enumerate(self._corr._base.inputs)
        )
        return f"correctionlib Correction: {self._corr.name}({signature})"
