from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from coffea.processor import ProcessorABC


def apply_to_one_dataset(
    proc: ProcessorABC, dataset, schemaclass=NanoAODSchema, metadata={}
):
    files = dataset["files"]
    events = NanoEventsFactory.from_root(
        files,
        metadata=metadata,
        schemaclass=schemaclass,
    ).events()
    return proc.process(events)


def apply_to_fileset(proc: ProcessorABC, fileset, schemaclass=NanoAODSchema):
    out = {}
    for name, dataset in fileset.items():
        metadata = dataset.get("metadata", {}).copy()
        metadata["dataset"] = name
        out[name] = apply_to_one_dataset(proc, dataset, schemaclass, metadata)
    return out
