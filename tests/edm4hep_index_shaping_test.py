from coffea.nanoevents import EDM4HEPSchema, NanoEventsFactory

events = NanoEventsFactory.from_root(
    "../../root_files_may18/rv02-02.sv02-02.mILD_l5_o1_v02.E250-SetA.I402004"
    ".Pe2e2h.eR.pL.n000.d_dstm_15090_*.slcio.edm4hep.root",
    treepath="events",
    schemaclass=EDM4HEPSchema,
    permit_dask=True,
    metadata={"b_field": 5},
).events()


print(events.RecoMCTruthLink.debug_index_shaping.compute(scheduler="sync"))
