from coffea.nanoevents import NanoAODSchema, NanoEventsFactory


def test_explicit_delete_after_assign():
    events = NanoEventsFactory.from_root(
        "https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dy.root",
        metadata={"dataset": "nano_dy"},
        schemaclass=NanoAODSchema,
        permit_dask=True,
    ).events()
    genpart = events["GenPart"]
    del events
    parent = genpart.parent  # noqa: F841
