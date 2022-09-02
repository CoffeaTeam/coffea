import awkward as ak
from coffea import processor, nanoevents
import hist
from collections import defaultdict


class NanoEventsProcessor(processor.ProcessorABC):
    def __init__(self, columns=[], canaries=[]):
        self._columns = columns
        self._canaries = canaries

        self.expected_usermeta = {
            "ZJets": ("someusermeta", "hello"),
            "Data": ("someusermeta2", "world"),
        }

    @property
    def columns(self):
        return self._columns

    @property
    def accumulator(self):
        dataset_axis = hist.axis.StrCategory(
            [], growth=True, name="dataset", label="Primary dataset"
        )
        mass_axis = hist.axis.Regular(
            30000, 0.25, 300, name="mass", label=r"$m_{\mu\mu}$ [GeV]"
        )
        pt_axis = hist.axis.Regular(30000, 0.24, 300, name="pt", label=r"$p_{T}$ [GeV]")

        accumulator = {
            # replace when py3.6 is dropped
            # "mass": hist.Hist(dataset_axis, mass_axis, name="Counts"),
            # "pt": hist.Hist(dataset_axis, pt_axis, name="Counts"),
            "mass": hist.Hist(dataset_axis, mass_axis),
            "pt": hist.Hist(dataset_axis, pt_axis),
            "cutflow": defaultdict(int),
            "worker": set(),
        }

        return accumulator

    def process(self, events):
        output = self.accumulator

        dataset = events.metadata["dataset"]
        print(events.metadata)
        if "checkusermeta" in events.metadata:
            metaname, metavalue = self.expected_usermeta[dataset]
            assert metavalue == events.metadata[metaname]

        mapping = events.behavior["__events_factory__"]._mapping
        muon_pt = events.Muon.pt
        if isinstance(mapping, nanoevents.mapping.CachedMapping):
            keys_in_cache = list(mapping.cache.cache.keys())
            has_canaries = [canary in keys_in_cache for canary in self._canaries]
            if has_canaries:
                try:
                    from distributed import get_worker

                    worker = get_worker()
                    output["worker"].add(worker.name)
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
