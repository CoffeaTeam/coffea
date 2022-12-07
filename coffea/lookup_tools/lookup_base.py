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

        def getfunction(args, **kwargs):
            if not isinstance(args, (list, tuple)):
                args = (args,)
            if all(
                isinstance(
                    x, (awkward.contents.NumpyArray, awkward.contents.EmptyArray)
                )
                or not isinstance(x, (awkward.contents.Content))
                for x in args
            ):
                nplike = awkward._nplikes.nplike_of(*args)
                if not isinstance(nplike, awkward._nplikes.Numpy):
                    raise NotImplementedError(
                        "support for cupy/jax/etc. numpy extensions"
                    )
                result = self._evaluate(*[awkward.to_numpy(x) for x in args], **kwargs)
                return awkward.contents.NumpyArray(result)
            return None

        behavior = awkward._util.behavior_of(*args)
        out = awkward.transform(getfunction, *args, behavior=behavior)
        return out

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError
