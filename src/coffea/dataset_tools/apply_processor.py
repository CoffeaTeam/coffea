from __future__ import annotations

import copy
from functools import partial
from typing import Any, Callable, Dict, Hashable, List, Set, Tuple, Union

import awkward
import dask.base
import dask.delayed
import dask_awkward

from coffea.dataset_tools.preprocess import (
    DatasetSpec,
    DatasetSpecOptional,
    FilesetSpec,
    FilesetSpecOptional,
)
from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from coffea.processor import ProcessorABC
from coffea.util import decompress_form

DaskOutputBaseType = Union[
    dask.base.DaskMethodsMixin,
    Dict[Hashable, dask.base.DaskMethodsMixin],
    Set[dask.base.DaskMethodsMixin],
    List[dask.base.DaskMethodsMixin],
    Tuple[dask.base.DaskMethodsMixin],
]

# NOTE TO USERS: You can use nested python containers as arguments to dask.compute!
DaskOutputType = Union[DaskOutputBaseType, Tuple[DaskOutputBaseType, ...]]

GenericHEPAnalysis = Callable[[dask_awkward.Array], DaskOutputType]


def _pack_meta_to_wire(*collections):
    unpacked, repacker = dask.base.unpack_collections(*collections)

    output = []
    for i in range(len(unpacked)):
        output.append(unpacked[i])
        if isinstance(
            unpacked[i], (dask_awkward.Array, dask_awkward.Record, dask_awkward.Scalar)
        ):
            output[-1]._meta = awkward.Array(
                unpacked[i]._meta.layout.form.length_zero_array(),
                behavior=unpacked[i]._meta.behavior,
                attrs=unpacked[i]._meta.attrs,
            )
    packed_out = repacker(output)
    if len(packed_out) == 1:
        return packed_out[0]
    return packed_out


def _unpack_meta_from_wire(*collections):
    unpacked, repacker = dask.base.unpack_collections(*collections)

    output = []
    for i in range(len(unpacked)):
        output.append(unpacked[i])
        if isinstance(
            unpacked[i], (dask_awkward.Array, dask_awkward.Record, dask_awkward.Scalar)
        ):
            output[-1]._meta = awkward.Array(
                unpacked[i]._meta.layout.to_typetracer(forget_length=True),
                behavior=unpacked[i]._meta.behavior,
                attrs=unpacked[i]._meta.attrs,
            )
    packed_out = repacker(output)
    if len(packed_out) == 1:
        return packed_out[0]
    return packed_out


def _apply_analysis_wire(analysis, events_and_maybe_report_wire):
    events = _unpack_meta_from_wire(events_and_maybe_report_wire)
    report = None
    if isinstance(events, tuple):
        events, report = events
    events._meta.attrs["@original_array"] = events

    out = analysis(events)
    if report is not None:
        return _pack_meta_to_wire(out, report)
    return _pack_meta_to_wire(out)


def apply_to_dataset(
    data_manipulation: ProcessorABC | GenericHEPAnalysis,
    dataset: DatasetSpec | DatasetSpecOptional,
    schemaclass: BaseSchema = NanoAODSchema,
    metadata: dict[Hashable, Any] = {},
    uproot_options: dict[str, Any] = {},
    parallelize_with_dask: bool = False,
) -> DaskOutputType | tuple[DaskOutputType, dask_awkward.Array]:
    """
    Apply the supplied function or processor to the supplied dataset.
    Parameters
    ----------
        data_manipulation : ProcessorABC or GenericHEPAnalysis
            The user analysis code to run on the input dataset
        dataset: DatasetSpec | DatasetSpecOptional
            The data to be acted upon by the data manipulation passed in.
        schemaclass: BaseSchema, default NanoAODSchema
            The nanoevents schema to interpret the input dataset with.
        metadata: dict[Hashable, Any], default {}
            Metadata for the dataset that is accessible by the input analysis. Should also be dask-serializable.
        uproot_options: dict[str, Any], default {}
            Options to pass to uproot. Pass at least {"allow_read_errors_with_report": True} to turn on file access reports.
        parallelize_with_dask: bool, default False
            Create dask.delayed objects that will return the the computable dask collections for the analysis when computed.

    Returns
    -------
        out : DaskOutputType
            The output of the analysis workflow applied to the dataset
        report : dask_awkward.Array, optional
            The file access report for running the analysis on the input dataset. Needs to be computed in simultaneously with the analysis to be accurate.
    """
    maybe_base_form = dataset.get("form", None)
    if maybe_base_form is not None:
        maybe_base_form = awkward.forms.from_json(decompress_form(maybe_base_form))
    files = dataset["files"]
    events_and_maybe_report = NanoEventsFactory.from_root(
        files,
        metadata=metadata,
        schemaclass=schemaclass,
        known_base_form=maybe_base_form,
        uproot_options=uproot_options,
    ).events()

    events = events_and_maybe_report
    report = None
    if isinstance(events, tuple):
        events, report = events

    analysis = None
    if isinstance(data_manipulation, ProcessorABC):
        analysis = data_manipulation.process
    elif isinstance(data_manipulation, Callable):
        analysis = data_manipulation
    else:
        raise ValueError("data_manipulation must either be a ProcessorABC or Callable")

    out = None
    if parallelize_with_dask:
        if not isinstance(events_and_maybe_report, tuple):
            events_and_maybe_report = (events_and_maybe_report,)
        wired_events = _pack_meta_to_wire(*events_and_maybe_report)
        out = dask.delayed(partial(_apply_analysis_wire, analysis, wired_events))()
    else:
        out = analysis(events)

    if report is not None:
        return out, report
    return (out,)


def apply_to_fileset(
    data_manipulation: ProcessorABC | GenericHEPAnalysis,
    fileset: FilesetSpec | FilesetSpecOptional,
    schemaclass: BaseSchema = NanoAODSchema,
    uproot_options: dict[str, Any] = {},
    parallelize_with_dask: bool = False,
    scheduler: Callable | str | None = None,
) -> dict[str, DaskOutputType] | tuple[dict[str, DaskOutputType], dask_awkward.Array]:
    """
    Apply the supplied function or processor to the supplied fileset (set of datasets).
    Parameters
    ----------
        data_manipulation : ProcessorABC or GenericHEPAnalysis
            The user analysis code to run on the input dataset
        fileset: FilesetSpec | FilesetSpecOptional
            The data to be acted upon by the data manipulation passed in. Metadata within the fileset should be dask-serializable.
        schemaclass: BaseSchema, default NanoAODSchema
            The nanoevents schema to interpret the input dataset with.
        uproot_options: dict[str, Any], default {}
            Options to pass to uproot. Pass at least {"allow_read_errors_with_report": True} to turn on file access reports.
        parallelize_with_dask: bool, default False
            Create dask.delayed objects that will return the the computable dask collections for the analysis when computed.
        scheduler: Callable | str | None, default None
            If parallelize_with_dask is True, this specifies the dask scheduler used to calculate task graphs in parallel.

    Returns
    -------
        out : dict[str, DaskOutputType]
            The output of the analysis workflow applied to the datasets, keyed by dataset name.
        report : dask_awkward.Array, optional
            The file access report for running the analysis on the input dataset. Needs to be computed in simultaneously with the analysis to be accurate.
    """
    out = {}
    analyses_to_compute = {}
    report = {}
    for name, dataset in fileset.items():
        metadata = copy.deepcopy(dataset.get("metadata", {}))
        if metadata is None:
            metadata = {}
        metadata.setdefault("dataset", name)
        dataset_out = apply_to_dataset(
            data_manipulation,
            dataset,
            schemaclass,
            metadata,
            uproot_options,
            parallelize_with_dask,
        )

        if parallelize_with_dask:
            analyses_to_compute[name] = dataset_out
        elif isinstance(dataset_out, tuple):
            out[name], report[name] = dataset_out
        else:
            out[name] = dataset_out[0]

    if parallelize_with_dask:
        (calculated_graphs,) = dask.compute(analyses_to_compute, scheduler=scheduler)
        for name, dataset_out_wire in calculated_graphs.items():
            to_unwire = dataset_out_wire
            if not isinstance(dataset_out_wire, tuple):
                to_unwire = (dataset_out_wire,)
            dataset_out = _unpack_meta_from_wire(*to_unwire)
            if isinstance(dataset_out, tuple):
                out[name], report[name] = dataset_out
            else:
                out[name] = dataset_out[0]

    if len(report) > 0:
        return out, report
    return out
