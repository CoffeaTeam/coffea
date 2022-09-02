from coffea import processor
import hist
import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector
from collections import defaultdict


class NanoTestProcessor(processor.ProcessorABC):
    def __init__(self, columns=[]):
        self._columns = columns
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
        }

        return accumulator

    def process(self, df):
        ak.behavior.update(vector.behavior)
        output = self.accumulator

        dataset = df.metadata["dataset"]
        if "checkusermeta" in df.metadata:
            metaname, metavalue = self.expected_usermeta[dataset]
            assert metavalue == df.metadata[metaname]

        muon = ak.zip(
            {
                "pt": df.Muon_pt,
                "eta": df.Muon_eta,
                "phi": df.Muon_phi,
                "mass": df.Muon_mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )

        dimuon = ak.combinations(muon, 2)
        dimuon = dimuon["0"] + dimuon["1"]

        output["pt"].fill(dataset=dataset, pt=ak.flatten(muon.pt))
        output["mass"].fill(dataset=dataset, mass=ak.flatten(dimuon.mass))
        output["cutflow"]["%s_pt" % dataset] += np.sum(ak.num(muon))
        output["cutflow"]["%s_mass" % dataset] += np.sum(ak.num(dimuon))

        return output

    def postprocess(self, accumulator):
        return accumulator
