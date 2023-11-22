import copy
from typing import Callable, Union

from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.processor import ProcessorABC


def apply_to_one_dataset(
    data_manipulation: Union[ProcessorABC, Callable],
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
    data_manipulation: Union[ProcessorABC, Callable], fileset, schemaclass=NanoAODSchema
):
    out = {}
    for name, dataset in fileset.items():
        metadata = copy.deepcopy(dataset.get("metadata", {}))
        metadata["dataset"] = name
        out[name] = apply_to_one_dataset(
            data_manipulation, dataset, schemaclass, metadata
        )
    return out
