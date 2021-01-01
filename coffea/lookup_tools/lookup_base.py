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

        def getfunction(inputs, depth):
            if all(
                isinstance(x, awkward.layout.NumpyArray)
                or not isinstance(
                    x, (awkward.layout.Content, awkward.partition.PartitionedArray)
                )
                for x in inputs
            ):
                nplike = awkward.nplike.of(*inputs)
                if not isinstance(nplike, awkward.nplike.Numpy):
                    raise NotImplementedError(
                        "support for cupy/jax/etc. numpy extensions"
                    )
                result = self._evaluate(*[nplike.asarray(x) for x in inputs], **kwargs)
                return lambda: (awkward.layout.NumpyArray(result),)
            return None

        behavior = awkward._util.behaviorof(*args)
        args = [
            awkward.operations.convert.to_layout(
                arg, allow_record=False, allow_other=True
            )
            for arg in args
        ]
        out = awkward._util.broadcast_and_apply(args, getfunction, behavior)
        assert isinstance(out, tuple) and len(out) == 1
        return awkward._util.wrap(out[0], behavior=behavior)

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError
