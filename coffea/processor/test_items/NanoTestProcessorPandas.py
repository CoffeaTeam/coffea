from coffea import hist, processor
from coffea.analysis_objects import JaggedCandidateArray as CandArray
from coffea.util import awkward as akd
from coffea.util import numpy as np
import pandas as pd


class NanoTestProcessorPandas(processor.ProcessorABC):
    def __init__(self, columns=[]):
        self._columns = columns

    @property
    def columns(self):
        return self._columns

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = pd.DataFrame()

        df = df[(df.Muon.pt > 20).any()]

        output['run'] = df.run.flatten()
        output['event'] = df.event.flatten()
        output['dataset'] = df.metadata['dataset']

        output['mu1_pt'] = -999.0
        output['mu2_pt'] = -999.0

        muons = df.Muon[df.Muon.pt > 20]

        one_muon = muons.counts > 0
        two_muons = muons.counts > 1

        output.loc[one_muon, 'mu1_pt'] = muons[one_muon][:, 0].pt.flatten()
        output.loc[two_muons, 'mu2_pt'] = muons[two_muons][:, 1].pt.flatten()
        return output

    def postprocess(self, accumulator):
        return accumulator
