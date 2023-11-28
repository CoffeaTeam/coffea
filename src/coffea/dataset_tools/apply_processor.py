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

DaskOutputType = Tuple[
    Union[
        dask.base.DaskMethodsMixin,
        Dict[Hashable, dask.base.DaskMethodsMixin],
        Set[dask.base.DaskMethodsMixin],
        List[dask.base.DaskMethodsMixin],
        Tuple[dask.base.DaskMethodsMixin],
    ],
    ...,
]  # NOTE TO USERS: You can use nested python containers as arguments to dask.compute!

GenericHEPAnalysis = Callable[[dask_awkward.Array], DaskOutputType]


def apply_to_dataset(
    data_manipulation: ProcessorABC | GenericHEPAnalysis,
    dataset: DatasetSpec | DatasetSpecOptional,
    schemaclass: BaseSchema = NanoAODSchema,
    metadata: dict[Hashable, Any] = {},
) -> DaskOutputType:
    files = dataset["files"]
    events = NanoEventsFactory.from_root(
        files,
        metadata=metadata,
        schemaclass=schemaclass,
    ).events()
    if isinstance(data_manipulation, ProcessorABC):
        return data_manipulation.process(events)
    elif isinstance(data_manipulation, Callable):
        return data_manipulation(events)
    else:
        raise ValueError("data_manipulation must either be a ProcessorABC or Callable")


def apply_to_fileset(
    data_manipulation: ProcessorABC | GenericHEPAnalysis,
    fileset: FilesetSpec | FilesetSpecOptional,
    schemaclass: BaseSchema = NanoAODSchema,
) -> dict[str, DaskOutputType]:
    out = {}
    for name, dataset in fileset.items():
        metadata = copy.deepcopy(dataset.get("metadata", {}))
        metadata.setdefault("dataset", name)
        out[name] = apply_to_dataset(data_manipulation, dataset, schemaclass, metadata)
    return out
