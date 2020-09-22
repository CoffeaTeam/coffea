import awkward1 as ak
from coffea import hist, processor


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

        dimuon = ak.combinations(events.Muon, 2)
        dimuon = dimuon["0"] + dimuon["1"]

        output["pt"].fill(dataset=dataset, pt=ak.flatten(events.Muon.pt))
        output["mass"].fill(dataset=dataset, mass=ak.flatten(dimuon.mass))
        output["cutflow"]["%s_pt" % dataset] += sum(ak.num(events.Muon))
        output["cutflow"]["%s_mass" % dataset] += sum(ak.num(dimuon))

        return output

    def postprocess(self, accumulator):
        return accumulator
