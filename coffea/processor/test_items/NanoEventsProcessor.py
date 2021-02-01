import awkward as ak
from coffea import hist, processor
from coffea import nanoevents


class NanoEventsProcessor(processor.ProcessorABC):
    def __init__(self, columns=[], canaries=[]):
        self._columns = columns
        self._canaries = canaries
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        mass_axis = hist.Bin("mass", r"$m_{\mu\mu}$ [GeV]", 30000, 0.25, 300)
        pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 30000, 0.25, 300)

        self._accumulator = processor.dict_accumulator(
            {
                "mass": hist.Hist("Counts", dataset_axis, mass_axis),
                "pt": hist.Hist("Counts", dataset_axis, pt_axis),
                "cutflow": processor.defaultdict_accumulator(int),
                "worker": processor.set_accumulator(),
            }
        )

    @property
    def columns(self):
        return self._columns

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        output = self.accumulator.identity()

        dataset = events.metadata["dataset"]

        mapping = events.behavior["__events_factory__"]._mapping
        muon_pt = events.Muon.pt
        if isinstance(mapping, nanoevents.mapping.CachedMapping):
            keys_in_cache = list(mapping.cache.cache.keys())
            has_canaries = [canary in keys_in_cache for canary in self._canaries]
            if has_canaries:
                try:
                    from distributed import get_worker
                    worker = get_worker()
                    output['worker'].add(worker.name)
                except ValueError:
                    pass

        dimuon = ak.combinations(events.Muon, 2)
        dimuon = dimuon["0"] + dimuon["1"]

        output["pt"].fill(dataset=dataset, pt=ak.flatten(muon_pt))
        output["mass"].fill(dataset=dataset, mass=ak.flatten(dimuon.mass))
        output["cutflow"]["%s_pt" % dataset] += sum(ak.num(events.Muon))
        output["cutflow"]["%s_mass" % dataset] += sum(ak.num(dimuon))

        return output

    def postprocess(self, accumulator):
        return accumulator
