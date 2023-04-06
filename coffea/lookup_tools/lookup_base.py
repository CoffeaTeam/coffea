import numbers
import weakref
from functools import partial

import awkward
import dask_awkward
import numpy


def getfunction(
    args, thelookup_dask=None, thelookup_wref=None, __pre_args__=tuple(), **kwargs
):
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
            thelookup = None
            if thelookup_wref is not None:
                thelookup = thelookup_wref()
            else:
                thelookup = thelookup_dask.compute()
            result = thelookup._evaluate(
                *(list(__pre_args__) + [awkward.to_numpy(arg) for arg in args]),
                **kwargs,
            )
        elif backend == "typetracer":
            zlargs = []
            for arg in args:
                arg._touch_data(recursive=True)
                zlargs.append(arg.form.length_zero_array())
            result = thelookup_wref()._evaluate(
                *(list(__pre_args__) + [awkward.to_numpy(zlarg) for zlarg in zlargs]),
                **kwargs,
            )
        else:
            raise NotImplementedError("support for cupy/jax/etc. numpy extensions")

        out = awkward.contents.NumpyArray(result)
        if backend == "typetracer":
            return out.to_typetracer(forget_length=True)
        return out
    return None


class _LookupXformFn:
    def __init__(self, *args, thelookup_dask, thelookup_wref, **kwargs):
        self.func = partial(
            getfunction,
            thelookup_dask=thelookup_dask,
            thelookup_wref=thelookup_wref,
            __pre_args__=args,
            **kwargs,
        )

    def __call__(self, *args):
        return awkward.transform(self.func, *args)


class lookup_base:
    """Base class for all objects that do some sort of value or function lookup"""

    def __init__(self, dask_future):
        self._dask_future = dask_future
        self._weakref = weakref.ref(self)

    def __call__(self, *args, **kwargs):
        dask_label = kwargs.pop("dask_label", None)
        # if our inputs are all dask_awkward arrays, then we should map_partitions
        if any(isinstance(x, (dask_awkward.Array)) for x in args):
            import dask

            delay_args = tuple(
                arg for arg in args if not isinstance(arg, dask_awkward.Array)
            )
            actual_args = tuple(
                arg for arg in args if isinstance(arg, dask_awkward.Array)
            )
            tomap = _LookupXformFn(
                *delay_args,
                thelookup_dask=self._dask_future,
                thelookup_wref=self._weakref,
                **kwargs,
            )

            zlargs = [arg._meta.layout.form.length_zero_array() for arg in actual_args]
            zlout = tomap(*zlargs)
            meta = dask_awkward.typetracer_from_form(zlout.layout.form)

            if dask_label is not None:
                return dask_awkward.map_partitions(
                    tomap,
                    *actual_args,
                    label=dask_label,
                    token=dask.base.tokenize(self._dask_future.name, *args),
                    meta=meta,
                )
            else:
                return dask_awkward.map_partitions(
                    tomap,
                    *actual_args,
                    token=dask.base.tokenize(self._dask_future.name, *args),
                    meta=meta,
                )

        if all(isinstance(x, (numpy.ndarray, numbers.Number, str)) for x in args):
            return self._evaluate(*args, **kwargs)
        elif any(
            not isinstance(x, (awkward.highlevel.Array, numbers.Number, str))
            for x in args
        ):
            raise TypeError(
                "lookup base must receive high level awkward arrays,"
                " numpy arrays, strings, or numbers!"
            )

        # behavior = awkward._util.behavior_of(*args)
        non_array_args = tuple(
            arg for arg in args if not isinstance(arg, awkward.highlevel.Array)
        )
        array_args = tuple(
            arg for arg in args if isinstance(arg, awkward.highlevel.Array)
        )
        func = partial(
            getfunction,
            thelookup_dask=self._dask_future,
            thelookup_wref=self._weakref,
            __pre_args__=non_array_args,
            **kwargs,
        )
        out = awkward.transform(func, *array_args)
        return out

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError
