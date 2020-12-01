import numpy
import awkward
import awkward1
import numbers


class lookup_base(object):
    """Base class for all objects that do some sort of value or function lookup"""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
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
        offsets = None
        # TODO: check can use offsets (this should always be true for striped)
        # Alternatively we can just use starts and stops
        for i in range(len(inputs)):
            if isinstance(inputs[i], awkward.JaggedArray):
                if offsets is not None and offsets.base is not inputs[i].offsets.base:
                    if type(offsets) is int:
                        raise Exception(
                            "Do not mix JaggedArrays and numpy arrays when calling derived class of lookup_base"
                        )
                    elif (
                        type(offsets) is numpy.ndarray
                        and offsets.base is not inputs[i].offsets.base
                    ):
                        raise Exception(
                            "All input jagged arrays must have a common structure (offsets)!"
                        )
                offsets = inputs[i].offsets
                inputs[i] = inputs[i].content
            elif isinstance(inputs[i], numpy.ndarray):
                if offsets is not None:
                    if type(offsets) is numpy.ndarray:
                        raise Exception(
                            "do not mix JaggedArrays and numpy arrays when calling a derived class of lookup_base"
                        )
                offsets = -1
        retval = self._evaluate(*tuple(inputs), **kwargs)
        if offsets is not None and type(offsets) is not int:
            retval = awkward.JaggedArray.fromoffsets(offsets, retval)
        return retval

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError
