import numpy
import awkward
import numbers


class lookup_base(object):
    """Base class for all objects that do some sort of value or function lookup"""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        if all(isinstance(x, (numpy.ndarray, numbers.Number)) for x in args):
            return self._evaluate(*args, **kwargs)
        elif any(not isinstance(x, awkward.Array) for x in args):
            raise TypeError(
                "lookup base must receive high level awkward arrays,"
                " numpy arrays, or numbers!"
            )

        def getfunction(layout, **kwargs):
            if all(
                isinstance(x, awkward.contents.NumpyArray)
                or not isinstance(x, (awkward.contents.Content))
                for x in layout
            ):
                nplike = awkward.nplikes.nplike_of(*layout)
                if not isinstance(nplike, awkward.nplikes.Numpy):
                    raise NotImplementedError(
                        "support for cupy/jax/etc. numpy extensions"
                    )
                result = self._evaluate(*[nplike.asarray(x) for x in layout], **kwargs)
                return awkward.contents.NumpyArray(result)
            return None

        behavior = awkward._util.behavior_of(*args)
        args = [
            awkward.to_layout(arg, allow_record=False, allow_other=True) for arg in args
        ]
        out = awkward.transform(getfunction, *args, behavior=behavior)
        return out

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError
