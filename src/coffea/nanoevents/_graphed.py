"""The ``graphed`` arm of :class:`coffea.nanoevents.NanoEventsFactory`.

``mode="graphed"`` records an analysis into a ``graphed`` graph instead of materialising it, through
the :class:`GraphedNanoArray` subclass installed here. Cross-references come out as ordinary graph edges with both collections as operands, so a worker
needs nothing but the recorded graph to replay them.

This is the only module in coffea that imports ``graphed``, and only ``mode="graphed"`` imports it.
"""

from inspect import getattr_static

import awkward
import graphed
import uproot
from graphed.array import BoundMethod
from graphed.awkward import AwkwardBackend
from graphed.provenance import register_internal

from coffea.util import _DaskProperty

# a recorded op's provenance is the analyst's line, not the coffea frame that recorded it
register_internal("coffea")

#: names whose deferred meaning is a systematics axis, which graphed spells its own way
_REFUSED = frozenset({"_ensure_systematics", "add_systematic"})

_VARY_POINTER = "graphed mode does not carry coffea systematics; build the variations with graphed.vary(...)"


class _GraphedBoundMethod(BoundMethod):
    """A behavior method call on a graphed array, recorded as a ``method`` op."""

    __slots__ = ()

    def __call__(self, *args, **kwargs):
        if any(callable(arg) for arg in (*args, *kwargs.values())):
            raise NotImplementedError(
                f"{self._name}() was passed a callable, and graphed mode has no way to record "
                f"one; if it varies event data, spell the variation with graphed.vary(...)"
            )
        return super().__call__(*args, **kwargs)


# graphed>=0.0.7's AwkwardArray routes ak.<fn>(deferred) to gak; older releases lack it.
_GraphedAwkwardArray = getattr(graphed.awkward, "AwkwardArray", graphed.Array)


class GraphedNanoArray(_GraphedAwkwardArray):
    """The deferred NanoEvents surface, mirroring ``dask_awkward.Array.__getattr__``."""

    __slots__ = ()

    def __getattr__(self, name):
        # order matters: a record field shadows a behavior of the same name
        if name.startswith("__"):
            raise AttributeError(name)
        session = self.session
        form = session.form(self)
        if name in form.tt.fields:
            return session.record_op("field", [self], {"field": name})
        if name in _REFUSED:
            raise NotImplementedError(_VARY_POINTER)
        tracer = session.backend._with_behavior(form.tt)
        static = getattr_static(tracer, name, None)
        if getattr(static, "_no_dispatch", False) or isinstance(static, _DaskProperty):
            return static._dask_get(tracer, type(tracer), self)
        kind = session.backend.attribute_kind(form, name)
        if name.startswith("_") and (
            kind != "method" or static is getattr_static(awkward.Array, name, None)
        ):
            # ak.Array internals (_layout, ...) and display hooks (_repr_mimebundle_) are no data
            raise AttributeError(name)
        if kind == "method":
            return _GraphedBoundMethod(self, name)
        if kind == "introspection":
            return session.backend.introspect(form, name)
        return session.record_op("field", [self], {"field": name})


class GraphedNanoBackend(AwkwardBackend):
    """An ``AwkwardBackend`` whose sessions hand back :class:`GraphedNanoArray`."""

    def array_type(self):
        return GraphedNanoArray


def check_from_root(
    schemaclass, steps_per_file, uproot_options, entry_start=None, entry_stop=None
):
    """Refuse the ``from_root`` arguments and schemas graphed mode cannot honour, and return the
    uproot options to forward."""
    if entry_start is not None or entry_stop is not None:
        raise NotImplementedError(
            "graphed mode does not take entry_start/entry_stop; bound the entries in the files "
            'spec, {path: {"object_path": tree, "steps": [[start, stop]]}}'
        )
    if steps_per_file is not uproot._util.unset:
        raise NotImplementedError(
            "graphed mode does not partition files here; pass steps_per_file where the plan is "
            "built, graphed.aggregate_plan(..., steps_per_file=...)"
        )
    if uproot_options.get("allow_read_errors_with_report"):
        raise NotImplementedError(
            "graphed mode has no report tuple; failed partitions are dead-lettered and retried "
            "by graphed.checkpoint.run_resumable"
        )
    if not isinstance(schemaclass, type) or not vars(schemaclass).get(
        "__graphed_capable__"
    ):
        raise NotImplementedError(
            f"graphed mode needs a schema declaring __graphed_capable__ = True in its own class "
            f"body, which {schemaclass!r} does not"
        )
    # uproot.graphed refuses the report key by presence, even switched off
    return {
        key: value
        for key, value in uproot_options.items()
        if key != "allow_read_errors_with_report"
    }
