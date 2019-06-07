from __future__ import print_function

from coffea import lookup_tools
import uproot
from coffea.util import awkward
from coffea.util import numpy as np

def jetmet_evaluator():
    from coffea.lookup_tools import extractor
    extract = extractor()

    extract.finalize()

    return extract.make_evaluator()


def test_factorized_jet_corrector():
    from coffea.jetmet_tools import FactorizedJetCorrector

    evaluator = jetmet_evaluator()

def test_jet_resolution():
    from coffea.jetmet_tools import JetResolution

    evaluator = jetmet_evaluator()

def test_jet_correction_uncertainty():
    from coffea.jetmet_tools import JetCorrectionUncertainty

    evaluator = jetmet_evaluator()

def test_jet_resolution_sf():
    from coffea.jetmet_tools import JetResolutionScaleFactor

    evaluator = jetmet_evaluator()

def test_jet_transformer():
    from coffea.jetmet_tools import JetTransformer

    evaluator = jetmet_evaluator()
