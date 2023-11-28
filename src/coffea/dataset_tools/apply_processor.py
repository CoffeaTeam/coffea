import copy
from typing import Callable, Dict, Hashable, List, Set, Tuple, Union

import dask.base
import dask_awkward

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.processor import ProcessorABC

GenericHEPAnalysis = Callable[
    [dask_awkward.Array],
    Tuple[
        Union[
            dask.base.DaskMethodsMixin,
            Dict[Hashable, dask.base.DaskMethodsMixin],
            Set[dask.base.DaskMethodsMixin],
            List[dask.base.DaskMethodsMixin],
            Tuple[dask.base.DaskMethodsMixin],
        ],
        ...,
    ],  # NOTE TO USERS: You can use nested python containers as arguments to dask.compute!
]


def apply_to_dataset(
    data_manipulation: Union[ProcessorABC, GenericHEPAnalysis],
    dataset,
    schemaclass=NanoAODSchema,
    metadata={},
):
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
    data_manipulation: Union[ProcessorABC, GenericHEPAnalysis],
    fileset,
    schemaclass=NanoAODSchema,
):
    out = {}
    for name, dataset in fileset.items():
        metadata = copy.deepcopy(dataset.get("metadata", {}))
        metadata.setdefault("dataset", name)
        out[name] = apply_to_dataset(data_manipulation, dataset, schemaclass, metadata)
    return out
