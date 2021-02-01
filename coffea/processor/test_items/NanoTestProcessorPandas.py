from coffea import hist, processor
import awkward as ak
import numpy as np
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

    def process(self, events):
        output = pd.DataFrame()

        events = events[ak.any(events.Muon.pt > 20, axis=-1)]

        output['run'] = ak.to_numpy(events.run)
        output['event'] = ak.to_numpy(events.event)
        output['dataset'] = events.metadata['dataset']

        output['mu1_pt'] = -999.0
        output['mu2_pt'] = -999.0

        muons = events.Muon[events.Muon.pt > 20]

        counts = ak.num(muons)
        one_muon = ak.to_numpy(counts > 0)
        two_muons = ak.to_numpy(counts > 1)

        output.loc[one_muon, 'mu1_pt'] = ak.to_numpy(muons[one_muon][:, 0].pt)
        output.loc[two_muons, 'mu2_pt'] = ak.to_numpy(muons[two_muons][:, 1].pt)
        return output

    def postprocess(self, accumulator):
        return accumulator
