from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Hashable, List, Set, Tuple, Union

import dask.base
import dask_awkward

from coffea.dataset_tools.preprocess import (
    DatasetSpec,
    DatasetSpecOptional,
    FilesetSpec,
    FilesetSpecOptional,
)
from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from coffea.processor import ProcessorABC

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


def apply_to_dataset(
    data_manipulation: ProcessorABC | GenericHEPAnalysis,
    dataset: DatasetSpec | DatasetSpecOptional,
    schemaclass: BaseSchema = NanoAODSchema,
    metadata: dict[Hashable, Any] = {},
    uproot_options: dict[str, Any] = {},
) -> DaskOutputType:
    files = dataset["files"]
    events = NanoEventsFactory.from_root(
        files,
        metadata=metadata,
        schemaclass=schemaclass,
        uproot_options=uproot_options,
    ).events()

    report = None
    if isinstance(events, tuple):
        events, report = events

    out = None
    if isinstance(data_manipulation, ProcessorABC):
        out = data_manipulation.process(events)
    elif isinstance(data_manipulation, Callable):
        out = data_manipulation(events)
    else:
        raise ValueError("data_manipulation must either be a ProcessorABC or Callable")

    if report is not None:
        return out, report
    return out


def apply_to_fileset(
    data_manipulation: ProcessorABC | GenericHEPAnalysis,
    fileset: FilesetSpec | FilesetSpecOptional,
    schemaclass: BaseSchema = NanoAODSchema,
    uproot_options: dict[str, Any] = {},
) -> dict[str, DaskOutputType]:
    out = {}
    report = {}
    for name, dataset in fileset.items():
        metadata = copy.deepcopy(dataset.get("metadata", {}))
        metadata.setdefault("dataset", name)
        dataset_out = apply_to_dataset(
            data_manipulation, dataset, schemaclass, metadata, uproot_options
        )
        if isinstance(out, tuple):
            out[name], report[name] = dataset_out
        else:
            out[name] = dataset_out
    if len(report) > 0:
        return out, report
    return out
