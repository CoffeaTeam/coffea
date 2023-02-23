import numbers
from functools import partial

import awkward
import dask_awkward
import numpy


def getfunction(args, thelookup=None, **kwargs):
    if not isinstance(args, (list, tuple)):
        args = (args,)
    if all(
        isinstance(x, (awkward.contents.NumpyArray, awkward.contents.EmptyArray))
        or not isinstance(x, (awkward.contents.Content))
        for x in args
    ):
        result = None
        backend = awkward.backend(*args)
        if backend == "cpu":
            result = thelookup._evaluate(
                *[awkward.to_numpy(arg) for arg in args], **kwargs
            )
        elif backend == "typetracer":
            zlargs = tuple(arg.form.length_zero_array() for arg in args)
            result = thelookup._evaluate(
                *[awkward.to_numpy(zlarg) for zlarg in zlargs], **kwargs
            )
        else:
            raise NotImplementedError("support for cupy/jax/etc. numpy extensions")

        out = awkward.contents.NumpyArray(result)
        if backend == "typetracer":
            out = awkward.contents.NumpyArray(result)
            return dask_awkward.typetracer_from_form(out.form).layout
        return out
    return None


class _LookupXformFn:
    def __init__(self, thelookup, **kwargs):
        self.func = partial(getfunction, thelookup=thelookup, **kwargs)

    def __call__(self, *args):
        return awkward.transform(self.func, *args)


class lookup_base:
    """Base class for all objects that do some sort of value or function lookup"""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        dask_label = kwargs.pop("dask_label", None)
        # if our inputs are all dask_awkward arrays, then we should map_partitions
        if all(isinstance(x, (dask_awkward.Array)) for x in args):
            tomap = _LookupXformFn(self, **kwargs)

            zlargs = [arg._meta.layout.form.length_zero_array() for arg in args]
            zlout = tomap(*zlargs)
            meta = dask_awkward.typetracer_from_form(zlout.layout.form)

            if dask_label:
                return dask_awkward.map_partitions(
                    tomap,
                    *args,
                    label=dask_label,
                    meta=meta,
                )
            else:
                return dask_awkward.map_partitions(
                    tomap,
                    *args,
                    meta=meta,
                )

        if all(isinstance(x, (numpy.ndarray, numbers.Number)) for x in args):
            return self._evaluate(*args, **kwargs)
        elif any(not isinstance(x, awkward.highlevel.Array) for x in args):
            raise TypeError(
                "lookup base must receive high level awkward arrays,"
                " numpy arrays, or numbers!"
            )

        # behavior = awkward._util.behavior_of(*args)
        func = partial(getfunction, thelookup=self, **kwargs)
        out = awkward.transform(func, *args)
        return out

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError
