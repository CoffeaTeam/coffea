from coffea import hist, processor
from coffea.analysis_objects import JaggedCandidateArray as CandArray
from coffea.util import awkward as akd
from coffea.util import numpy as np


class NanoTestProcessor(processor.ProcessorABC):
    def __init__(self, columns=[]):
        self._columns = columns
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        mass_axis = hist.Bin("mass", r"$m_{\mu\mu}$ [GeV]", 30000, 0.25, 300)
        pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 30000, 0.25, 300)

        self._accumulator = processor.dict_accumulator({
                                                       'mass': hist.Hist("Counts", dataset_axis, mass_axis),
                                                       'pt': hist.Hist("Counts", dataset_axis, pt_axis),
                                                       'cutflow': processor.defaultdict_accumulator(int),
                                                       })

    @property
    def columns(self):
        return self._columns

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()

        dataset = df['dataset']

        muon = None
        if isinstance(df['Muon_pt'], akd.JaggedArray):
            muon = CandArray.candidatesfromcounts(counts=df['Muon_pt'].counts,
                                                  pt=df['Muon_pt'].content,
                                                  eta=df['Muon_eta'].content,
                                                  phi=df['Muon_phi'].content,
                                                  mass=df['Muon_mass'].content)
        else:
            muon = CandArray.candidatesfromcounts(counts=df['nMuon'],
                                                  pt=df['Muon_pt'],
                                                  eta=df['Muon_eta'],
                                                  phi=df['Muon_phi'],
                                                  mass=df['Muon_mass'])

        dimuon = muon.distincts()

        output['pt'].fill(dataset=dataset, pt=muon.pt.flatten())
        output['mass'].fill(dataset=dataset, mass=dimuon.mass.flatten())
        output['cutflow']['%s_pt' % dataset] += np.sum(muon.counts)
        output['cutflow']['%s_mass' % dataset] += np.sum(dimuon.counts)

        return output

    def postprocess(self, accumulator):
        return accumulator
