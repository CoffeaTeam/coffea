import numbers
from functools import partial

import awkward
import dask_awkward
import numpy


def getfunction(args, thelookup=None, __pre_args__=tuple(), **kwargs):
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
                *(list(__pre_args__) + [awkward.to_numpy(arg) for arg in args]),
                **kwargs,
            )
        elif backend == "typetracer":
            zlargs = tuple(arg.form.length_zero_array() for arg in args)
            result = thelookup._evaluate(
                *(list(__pre_args__) + [awkward.to_numpy(zlarg) for zlarg in zlargs]),
                **kwargs,
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
    def __init__(self, *args, thelookup, **kwargs):
        self.func = partial(
            getfunction, thelookup=thelookup, __pre_args__=args, **kwargs
        )

    def __call__(self, *args):
        return awkward.transform(self.func, *args)

    def __dask_tokenize__(self):
        return (_LookupXformFn, self.func)


class lookup_base:
    """Base class for all objects that do some sort of value or function lookup"""

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        dask_label = kwargs.pop("dask_label", None)
        # if our inputs are all dask_awkward arrays, then we should map_partitions
        if any(isinstance(x, (dask_awkward.Array)) for x in args):
            delay_args = tuple(
                arg for arg in args if not isinstance(arg, dask_awkward.Array)
            )
            actual_args = tuple(
                arg for arg in args if isinstance(arg, dask_awkward.Array)
            )
            tomap = _LookupXformFn(*delay_args, thelookup=self, **kwargs)

            zlargs = [arg._meta.layout.form.length_zero_array() for arg in actual_args]
            zlout = tomap(*zlargs)
            meta = dask_awkward.typetracer_from_form(zlout.layout.form)

            if dask_label:
                return dask_awkward.map_partitions(
                    tomap,
                    *actual_args,
                    label=dask_label,
                    meta=meta,
                )
            else:
                return dask_awkward.map_partitions(
                    tomap,
                    *actual_args,
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

    def __dask_tokenize__(self):
        return (lookup_base, self)

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError
