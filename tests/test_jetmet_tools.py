from __future__ import print_function
import sys

from coffea import lookup_tools
import uproot
from coffea.util import awkward
from coffea.util import numpy as np

import pytest

from dummy_distributions import dummy_jagged_eta_pt, dummy_four_momenta


def jetmet_evaluator():
    from coffea.lookup_tools import extractor
    extract = extractor()

    extract.add_weight_sets(['* * tests/samples/Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi.jec.txt.gz',
                             '* * tests/samples/Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi.jec.txt.gz',
                             '* * tests/samples/Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi.jec.txt.gz',
                             '* * tests/samples/Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi.jec.txt.gz',
                             '* * tests/samples/Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz',
                             '* * tests/samples/Summer16_23Sep2016V3_MC_Uncertainty_AK4PFPuppi.junc.txt.gz',
                             '* * tests/samples/Spring16_25nsV10_MC_PtResolution_AK4PFPuppi.jr.txt.gz',
                             '* * tests/samples/Spring16_25nsV10_MC_SF_AK4PFPuppi.jersf.txt.gz'])
    
    extract.finalize()

    return extract.make_evaluator()

evaluator = jetmet_evaluator()


def test_factorized_jet_corrector():
    from coffea.jetmet_tools import FactorizedJetCorrector

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    test_Rho = np.full_like(test_eta, 100.)
    test_A = np.full_like(test_eta, 5.)
    
    jec_names = ['Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi',
                 'Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi',
                 'Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi',
                 'Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi']
    corrector = FactorizedJetCorrector(**{name: evaluator[name] for name in jec_names})

    print(corrector)

    corrs = corrector.getCorrection(JetEta=test_eta, Rho=test_Rho, JetPt=test_pt, JetA=test_A)


def test_jet_resolution():
    from coffea.jetmet_tools import JetResolution

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    
    test_Rho = np.full_like(test_eta, 100.)
    
    jer_names = ['Spring16_25nsV10_MC_PtResolution_AK4PFPuppi']
    reso = JetResolution(**{name: evaluator[name] for name in jer_names})
                 
    print(reso)
                 
    resos = reso.getResolution(JetEta=test_eta, Rho=test_Rho, JetPt=test_pt)


def test_jet_correction_uncertainty():
    from coffea.jetmet_tools import JetCorrectionUncertainty

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    
    junc_names = ['Summer16_23Sep2016V3_MC_Uncertainty_AK4PFPuppi']
    junc = JetCorrectionUncertainty(**{name: evaluator[name] for name in junc_names})

    print(junc)

    juncs = junc.getUncertainty(JetEta=test_eta, JetPt=test_pt)

    for level, corrs in juncs:
        assert(corrs.shape[0] == test_eta.shape[0])


def test_jet_correction_uncertainty_sources():
    from coffea.jetmet_tools import JetCorrectionUncertainty

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    
    junc_names = []
    levels = []
    for name in dir(evaluator):
        if 'Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi' in name:
            junc_names.append(name)
            levels.append(name.split('_')[-1])
    junc = JetCorrectionUncertainty(**{name: evaluator[name] for name in junc_names})
    
    print(junc)

    juncs = junc.getUncertainty(JetEta=test_eta, JetPt=test_pt)

    for level, corrs in juncs:
        assert(level in levels)
        assert(corrs.shape[0] == test_eta.shape[0])


def test_jet_resolution_sf():
    from coffea.jetmet_tools import JetResolutionScaleFactor

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    
    jersf_names = ['Spring16_25nsV10_MC_SF_AK4PFPuppi']
    resosf = JetResolutionScaleFactor(**{name: evaluator[name] for name in jersf_names})
    
    print(resosf)
    
    resosfs = resosf.getScaleFactor(JetEta=test_eta)


def test_jet_transformer():
    from coffea.analysis_objects import JaggedCandidateArray as CandArray
    from coffea.jetmet_tools import (FactorizedJetCorrector,
                                     JetResolution,
                                     JetResolutionScaleFactor,
                                     JetCorrectionUncertainty,
                                     JetTransformer)
    
    counts, test_px, test_py, test_pz, test_e = dummy_four_momenta()
    
    test_Rho = np.full(shape=(np.sum(counts),), fill_value=100.)
    test_A = np.full(shape=(np.sum(counts),), fill_value=5.)

    jets = CandArray.candidatesfromcounts(counts, px=test_px, py=test_py, pz=test_pz, energy=test_e)
    jets.add_attributes(ptRaw=jets.pt,
                        massRaw=jets.mass,
                        rho=test_Rho,
                        area=test_A)
    
    jec_names = ['Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi',
                 'Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi',
                 'Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi',
                 'Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi']
    corrector = FactorizedJetCorrector(**{name: evaluator[name] for name in jec_names})
    
    junc_names = []
    for name in dir(evaluator):
        if 'Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi' in name:
            junc_names.append(name)
    junc = JetCorrectionUncertainty(**{name: evaluator[name] for name in junc_names})

    jer_names = ['Spring16_25nsV10_MC_PtResolution_AK4PFPuppi']
    reso = JetResolution(**{name: evaluator[name] for name in jer_names})
    
    jersf_names = ['Spring16_25nsV10_MC_SF_AK4PFPuppi']
    resosf = JetResolutionScaleFactor(**{name: evaluator[name] for name in jersf_names})

    xform = JetTransformer(jec=corrector, junc=junc, jer=reso, jersf=resosf)

    print(xform.uncertainties)

    xform.transform(jets)

    print(jets.columns)

    assert('pt_jer_up' in jets.columns)
    assert('pt_jer_down' in jets.columns)
    assert('mass_jer_up' in jets.columns)
    assert('mass_jer_down' in jets.columns)
    
    for unc in xform.uncertainties:
        assert('pt_'+unc+'_up' in jets.columns)
        assert('pt_'+unc+'_down' in jets.columns)
        assert('mass_'+unc+'_up' in jets.columns)
        assert('mass_'+unc+'_down' in jets.columns)
