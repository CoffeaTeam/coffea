import numbers
import weakref
from functools import partial

import awkward
import dask_awkward
import numpy


def getfunction(
    args,
    thelookup_dask=None,
    thelookup_wref=None,
    __non_array_args__=tuple(),
    __arg_indices__=tuple(),
    **kwargs,
):
    if not isinstance(args, (list, tuple)):
        args = (args,)

    if not len(args) + len(__non_array_args__) == len(__arg_indices__):
        raise ValueError(
            "Total length of array and non-array args should match expected placement."
        )

    if all(
        isinstance(x, (awkward.contents.NumpyArray, awkward.contents.EmptyArray))
        or not isinstance(x, (awkward.contents.Content))
        for x in args
    ):
        result = None
        backend = awkward.backend(*args)

        if backend != "cpu" and backend != "typetracer":
            raise NotImplementedError("support for cupy/jax/etc. numpy extensions")

        # order args and __non_array_args__ correctly
        repacked_args = [None] * len(__arg_indices__)
        for iarg, arg in enumerate(args):
            if backend == "cpu":
                repacked_args[__arg_indices__[iarg]] = awkward.to_numpy(arg)
            elif backend == "typetracer":
                repacked_args[__arg_indices__[iarg]] = awkward.to_numpy(
                    awkward.typetracer.length_zero_if_typetracer(arg)
                )

        for inaarg, naarg in enumerate(__non_array_args__):
            repacked_args[__arg_indices__[inaarg + len(args)]] = naarg

        thelookup = None
        if thelookup_wref is not None:
            thelookup = thelookup_wref()
        else:
            from dask.distributed import worker_client

            with worker_client() as client:
                thelookup = client.compute(thelookup_dask).result()

        result = thelookup._evaluate(*repacked_args, **kwargs)
        out = awkward.contents.NumpyArray(result)
        if backend == "typetracer":
            return out.to_typetracer(forget_length=True)
        return out
    return None


class _LookupXformFn:
    def __init__(self, *args, arg_indices, thelookup_dask, thelookup_wref, **kwargs):
        self.getfunction = getfunction
        self._thelookup_dask = thelookup_dask
        self._thelookup_wref = thelookup_wref
        self.__non_array_args__ = args
        self.__arg_indices__ = arg_indices
        self.kwargs = kwargs

    def __getstate__(self):
        out = self.__dict__.copy()
        out["_thelookup_wref"] = None
        return out

    def __call__(self, *args):
        func = partial(
            self.getfunction,
            thelookup_dask=self._thelookup_dask,
            thelookup_wref=self._thelookup_wref,
            __non_array_args__=self.__non_array_args__,
            __arg_indices__=self.__arg_indices__,
            **self.kwargs,
        )
        return awkward.transform(func, *args)


class lookup_base:
    """Base class for all objects that do some sort of value or function lookup"""

    def __init__(self, dask_future):
        self._dask_future = dask_future
        self._weakref = weakref.ref(self)

    def __getstate__(self):
        out = self.__dict__.copy()
        if "_weakref" in out:
            out["_weakref"] = None
        return out

    def __call__(self, *args, **kwargs):
        dask_label = kwargs.pop("dask_label", None)

        actual_args = []
        actual_arg_indices = []
        delay_args = []
        delay_arg_indices = []
        for iarg, arg in enumerate(args):
            if isinstance(arg, (awkward.highlevel.Array, dask_awkward.Array)):
                actual_args.append(arg)
                actual_arg_indices.append(iarg)
            else:
                delay_args.append(arg)
                delay_arg_indices.append(iarg)
        arg_indices = tuple(actual_arg_indices + delay_arg_indices)
        actual_args = tuple(actual_args)
        delay_args = tuple(delay_args)

        tomap = _LookupXformFn(
            *delay_args,
            arg_indices=arg_indices,
            thelookup_dask=self._dask_future,
            thelookup_wref=self._weakref,
            **kwargs,
        )

        # if our inputs are all dask_awkward arrays, then we should map_partitions
        if any(isinstance(x, (dask_awkward.Array)) for x in args):
            from dask.base import tokenize

            zlargs = [
                awkward.Array(
                    arg._meta.layout.form.length_zero_array(highlevel=False),
                    behavior=arg.behavior,
                )
                for arg in actual_args
            ]
            zlout = tomap(*zlargs)
            meta = awkward.Array(
                zlout.layout.to_typetracer(forget_length=True), behavior=zlout.behavior
            )

            if dask_label is not None:
                return dask_awkward.map_partitions(
                    tomap,
                    *actual_args,
                    label=dask_label,
                    token=tokenize(self._dask_future.name, *args),
                    meta=meta,
                )
            else:
                return dask_awkward.map_partitions(
                    tomap,
                    *actual_args,
                    token=tokenize(self._dask_future.name, *args),
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

        return tomap(*actual_args)

    def _evaluate(self, *args, **kwargs):
        raise NotImplementedError
