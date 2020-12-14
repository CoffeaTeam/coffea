import numpy
import awkward
import awkward1
import numbers

from coffea.util import deprecate_detected_awkward0


class lookup_base(object):
    """Base class for all objects that do some sort of value or function lookup"""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        deprecate_detected_awkward0(*args, **kwargs)
        if all(isinstance(x, (numpy.ndarray, numbers.Number)) for x in args):
            return self._evaluate(*args, **kwargs)
        elif any(isinstance(x, awkward.JaggedArray) for x in args):
            return self._call_ak0(list(args), **kwargs)

        def getfunction(inputs, depth):
            if all(
                isinstance(x, awkward1.layout.NumpyArray)
                or not isinstance(
                    x, (awkward1.layout.Content, awkward1.partition.PartitionedArray)
                )
                for x in inputs
            ):
                nplike = awkward1.nplike.of(*inputs)
                if not isinstance(nplike, awkward1.nplike.Numpy):
                    raise NotImplementedError(
                        "support for cupy/jax/etc. numpy extensions"
                    )
                result = self._evaluate(*[nplike.asarray(x) for x in inputs], **kwargs)
                return lambda: (awkward1.layout.NumpyArray(result),)
            return None

        behavior = awkward1._util.behaviorof(*args)
        args = [
            awkward1.operations.convert.to_layout(
                arg, allow_record=False, allow_other=True
            )
            for arg in args
        ]
        out = awkward1._util.broadcast_and_apply(args, getfunction, behavior)
        assert isinstance(out, tuple) and len(out) == 1
        return awkward1._util.wrap(out[0], behavior=behavior)

    def _call_ak0(self, inputs, **kwargs):
        if not all([isinstance(input, awkward.JaggedArray) for input in inputs]):
            raise Exception(
                "Do not mix JaggedArrays and other arrays when calling derived class of lookup_base"
            )
        wrap, arrays = awkward.util.unwrap_jagged(inputs[0],
                                                  inputs[0].JaggedArray,
                                                  inputs)
        return wrap(self._evaluate(*tuple(arrays), **kwargs))

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError
