from coffea import hist, processor
import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector


class NanoTestProcessor(processor.ProcessorABC):
    def __init__(self, columns=[]):
        self._columns = columns
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        mass_axis = hist.Bin("mass", r"$m_{\mu\mu}$ [GeV]", 30000, 0.25, 300)
        pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 30000, 0.25, 300)

        self._accumulator = processor.dict_accumulator(
            {
                "mass": hist.Hist("Counts", dataset_axis, mass_axis),
                "pt": hist.Hist("Counts", dataset_axis, pt_axis),
                "cutflow": processor.defaultdict_accumulator(int),
            }
        )

    @property
    def columns(self):
        return self._columns

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        ak.behavior.update(vector.behavior)
        output = self.accumulator.identity()

        dataset = df.metadata["dataset"]

        muon = ak.zip({'pt': df.Muon_pt,
                       'eta': df.Muon_eta,
                       'phi': df.Muon_phi,
                       'mass': df.Muon_mass},
                      with_name="PtEtaPhiMLorentzVector")

        dimuon = ak.combinations(muon, 2)
        dimuon = dimuon["0"] + dimuon["1"]

        output["pt"].fill(dataset=dataset, pt=ak.flatten(muon.pt))
        output["mass"].fill(dataset=dataset, mass=ak.flatten(dimuon.mass))
        output["cutflow"]["%s_pt" % dataset] += np.sum(ak.num(muon))
        output["cutflow"]["%s_mass" % dataset] += np.sum(ak.num(dimuon))

        return output

    def postprocess(self, accumulator):
        return accumulator
