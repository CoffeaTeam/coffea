from __future__ import print_function
import sys

from coffea import lookup_tools
import uproot
from coffea.util import awkward
from coffea.util import awkward1
from coffea.util import numpy as np

import pytest, time # , pyinstrument

from dummy_distributions import dummy_jagged_eta_pt, dummy_four_momenta

awkwardlibs = ["ak0", "ak1"]


def jetmet_evaluator():
    from coffea.lookup_tools import extractor
    extract = extractor()

    extract.add_weight_sets(['* * tests/samples/Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi.jec.txt.gz',
                             '* * tests/samples/Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi.jec.txt.gz',
                             '* * tests/samples/Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi.jec.txt.gz',
                             '* * tests/samples/Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi.jec.txt.gz',
                             '* * tests/samples/Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz',
                             '* * tests/samples/Summer16_23Sep2016V3_MC_Uncertainty_AK4PFPuppi.junc.txt.gz',
                             '* * tests/samples/Fall17_17Nov2017_V6_MC_UncertaintySources_AK4PFchs.junc.txt.gz',
                             '* * tests/samples/Regrouped_Fall17_17Nov2017_V32_MC_UncertaintySources_AK4PFchs.junc.txt',
                             '* * tests/samples/Spring16_25nsV10_MC_PtResolution_AK4PFPuppi.jr.txt.gz',
                             '* * tests/samples/Spring16_25nsV10_MC_SF_AK4PFPuppi.jersf.txt.gz',
                             '* * tests/samples/Autumn18_V7_MC_SF_AK4PFchs.jersf.txt.gz'])

    extract.finalize()

    return extract.make_evaluator()

evaluator = jetmet_evaluator()


@pytest.mark.parametrize("awkwardlib", awkwardlibs)
def test_factorized_jet_corrector(awkwardlib):
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

    pt_copy = np.copy(test_pt)

    corrs = corrector.getCorrection(JetEta=test_eta, Rho=test_Rho, JetPt=test_pt, JetA=test_A)

    assert((np.abs(pt_copy - test_pt) < 1e-6).all())

    # Test for bug #320
    def make_starts_stops(counts, data, padding):
        cumcounts = np.cumsum(counts)
        first_nonempty_event = cumcounts[np.nonzero(counts > 1)[0][0]]
        data_pad = np.empty(data.size + padding)
        data_pad[:first_nonempty_event] = data[:first_nonempty_event]
        data_pad[first_nonempty_event + padding:] = data[first_nonempty_event:]
        starts = np.r_[0, cumcounts[:-1]]
        starts[starts >= first_nonempty_event] += padding
        return awkward.JaggedArray(starts, starts + counts, data_pad)

    test_pt_jag = make_starts_stops(counts, test_pt, 5)
    test_eta_jag = make_starts_stops(counts, test_eta, 3)
    test_Rho_jag = awkward.JaggedArray.fromcounts(counts, test_Rho)
    test_A_jag = awkward.JaggedArray.fromcounts(counts, test_A)
    if awkwardlib == "ak1":
        test_pt_jag = awkward1.from_awkward0(test_pt_jag)
        test_eta_jag = awkward1.from_awkward0(test_eta_jag)
        test_Rho_jag = awkward1.from_awkward0(test_Rho_jag)
        test_A_jag = awkward1.from_awkward0(test_A_jag)

    corrs_jag = corrector.getCorrection(JetEta=test_eta_jag, Rho=test_Rho_jag, JetPt=test_pt_jag, JetA=test_A_jag)

    if awkwardlib == "ak1":
        assert(awkward1.all(np.abs(pt_copy - awkward1.flatten(test_pt_jag)) < 1e-6))
        assert(awkward1.all(np.abs(corrs - awkward1.flatten(corrs_jag)) < 1e-6))
    else:
        assert((np.abs(pt_copy - test_pt_jag.flatten()) < 1e-6).all())
        assert((np.abs(corrs - corrs_jag.flatten()) < 1e-6).all())


@pytest.mark.parametrize("awkwardlib", awkwardlibs)
def test_jet_resolution(awkwardlib):
    from coffea.jetmet_tools import JetResolution

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    test_Rho = np.full_like(test_eta, 100.)
    
    test_pt_jag = awkward.JaggedArray.fromcounts(counts, test_pt)
    test_eta_jag = awkward.JaggedArray.fromcounts(counts, test_eta)
    test_Rho_jag = awkward.JaggedArray.fromcounts(counts, test_Rho)
    if awkwardlib == "ak1":
        test_pt_jag = awkward1.from_awkward0(test_pt_jag)
        test_eta_jag = awkward1.from_awkward0(test_eta_jag)
        test_Rho_jag = awkward1.from_awkward0(test_Rho_jag)

    jer_names = ['Spring16_25nsV10_MC_PtResolution_AK4PFPuppi']
    reso = JetResolution(**{name: evaluator[name] for name in jer_names})

    print(reso)

    resos = reso.getResolution(JetEta=test_eta, Rho=test_Rho, JetPt=test_pt)
    
    resos = reso.getResolution(JetEta=test_eta_jag, Rho=test_Rho_jag, JetPt=test_pt_jag)


@pytest.mark.parametrize("awkwardlib", awkwardlibs)
def test_jet_correction_uncertainty(awkwardlib):
    from coffea.jetmet_tools import JetCorrectionUncertainty

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    test_pt_jag = awkward.JaggedArray.fromcounts(counts, test_pt)
    test_eta_jag = awkward.JaggedArray.fromcounts(counts, test_eta)
    if awkwardlib == "ak1":
        test_pt_jag = awkward1.from_awkward0(test_pt_jag)
        test_eta_jag = awkward1.from_awkward0(test_eta_jag)

    junc_names = ['Summer16_23Sep2016V3_MC_Uncertainty_AK4PFPuppi']
    junc = JetCorrectionUncertainty(**{name: evaluator[name] for name in junc_names})

    print(junc)

    juncs = junc.getUncertainty(JetEta=test_eta, JetPt=test_pt)

    juncs_jag = list(junc.getUncertainty(JetEta=test_eta_jag, JetPt=test_pt_jag))
    
    for i, (level, corrs) in enumerate(juncs):
        assert(corrs.shape[0] == test_eta.shape[0])
        if awkwardlib == "ak1":
            assert(awkward1.all(corrs == awkward1.flatten(juncs_jag[i][1])))
        if awkwardlib == "ak0":
            assert((corrs == juncs_jag[i][1].flatten()).all())


@pytest.mark.parametrize("awkwardlib", awkwardlibs)
def test_jet_correction_uncertainty_sources(awkwardlib):
    from coffea.jetmet_tools import JetCorrectionUncertainty

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    test_pt_jag = awkward.JaggedArray.fromcounts(counts, test_pt)
    test_eta_jag = awkward.JaggedArray.fromcounts(counts, test_eta)
    if awkwardlib == "ak1":
        test_pt_jag = awkward1.from_awkward0(test_pt_jag)
        test_eta_jag = awkward1.from_awkward0(test_eta_jag)

    junc_names = []
    levels = []
    for name in dir(evaluator):
        if 'Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi' in name:
            junc_names.append(name)
            levels.append(name.split('_')[-1])
        #test for underscore in dataera
        if 'Fall17_17Nov2017_V6_MC_UncertaintySources_AK4PFchs_AbsoluteFlavMap' in name:
            junc_names.append(name)
            levels.append(name.split('_')[-1])
    junc = JetCorrectionUncertainty(**{name: evaluator[name] for name in junc_names})

    print(junc)

    juncs = junc.getUncertainty(JetEta=test_eta, JetPt=test_pt)
    
    juncs_jag = list(junc.getUncertainty(JetEta=test_eta_jag, JetPt=test_pt_jag))

    for i, (level, corrs) in enumerate(juncs):
        assert(level in levels)
        assert(corrs.shape[0] == test_eta.shape[0])
        tic = time.time()
        if awkwardlib == "ak1":
            assert(awkward1.all(corrs == awkward1.flatten(juncs_jag[i][1])))
        if awkwardlib == "ak0":
            assert((corrs == juncs_jag[i][1].flatten()).all())
        toc = time.time()


@pytest.mark.parametrize("awkwardlib", awkwardlibs)
def test_jet_correction_regrouped_uncertainty_sources(awkwardlib):
    from coffea.jetmet_tools import JetCorrectionUncertainty

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    test_pt_jag = awkward.JaggedArray.fromcounts(counts, test_pt)
    test_eta_jag = awkward.JaggedArray.fromcounts(counts, test_eta)
    if awkwardlib == "ak1":
        test_pt_jag = awkward1.from_awkward0(test_pt_jag)
        test_eta_jag = awkward1.from_awkward0(test_eta_jag)

    junc_names = []
    levels = []
    for name in dir(evaluator):
        if 'Regrouped_Fall17_17Nov2017_V32_MC_UncertaintySources_AK4PFchs' in name:
            junc_names.append(name)
            if len(name.split('_')) == 9:
                levels.append("_".join(name.split('_')[-2:]))
            else:
                levels.append(name.split('_')[-1])
    junc = JetCorrectionUncertainty(**{name: evaluator[name] for name in junc_names})

    print(junc)

    juncs_jag = list(junc.getUncertainty(JetEta=test_eta_jag, JetPt=test_pt_jag))

    for i, tpl in enumerate(list(junc.getUncertainty(JetEta=test_eta, JetPt=test_pt))):
        assert(tpl[0] in levels)
        assert(tpl[1].shape[0] == test_eta.shape[0])
        if awkwardlib == "ak1":
            assert(awkward1.all(tpl[1] == awkward1.flatten(juncs_jag[i][1])))
        if awkwardlib == "ak0":
            assert((tpl[1] == juncs_jag[i][1].flatten()).all())

@pytest.mark.parametrize("awkwardlib", awkwardlibs)
def test_jet_resolution_sf(awkwardlib):
    from coffea.jetmet_tools import JetResolutionScaleFactor

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    test_pt_jag = awkward.JaggedArray.fromcounts(counts, test_pt)
    test_eta_jag = awkward.JaggedArray.fromcounts(counts, test_eta)
    if awkwardlib == "ak1":
        test_pt_jag = awkward1.from_awkward0(test_pt_jag)
        test_eta_jag = awkward1.from_awkward0(test_eta_jag)

    jersf_names = ['Spring16_25nsV10_MC_SF_AK4PFPuppi']
    resosf = JetResolutionScaleFactor(**{name: evaluator[name] for name in jersf_names})

    print(resosf)

    resosfs = resosf.getScaleFactor(JetEta=test_eta)
    
    resosfs_jag = resosf.getScaleFactor(JetEta=test_eta_jag)
    
    if awkwardlib == "ak1":
        assert(awkward1.all(resosfs == awkward1.flatten(resosfs_jag)))
    if awkwardlib == "ak0":
        assert((resosfs == resosfs_jag.flatten()).all())

@pytest.mark.parametrize("awkwardlib", awkwardlibs)
def test_jet_resolution_sf_2d(awkwardlib):
    from coffea.jetmet_tools import JetResolutionScaleFactor
    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    
    test_pt_jag = awkward.JaggedArray.fromcounts(counts, test_pt)
    test_eta_jag = awkward.JaggedArray.fromcounts(counts, test_eta)
    if awkwardlib == "ak1":
        test_pt_jag = awkward1.from_awkward0(test_pt_jag)
        test_eta_jag = awkward1.from_awkward0(test_eta_jag)
    
    resosf = JetResolutionScaleFactor(**{name: evaluator[name] for name in ["Autumn18_V7_MC_SF_AK4PFchs"]})
    
    resosfs = resosf.getScaleFactor(JetPt=test_pt, JetEta=test_eta)
    
    resosfs_jag = resosf.getScaleFactor(JetPt=test_pt_jag, JetEta=test_eta_jag)
    

def test_jet_transformer():
    import numpy as np
    import awkward as ak
    import math
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

    fakemet = np.random.exponential(scale=1.0,size=counts.size)
    metphi = np.random.uniform(low=-math.pi, high=math.pi, size=counts.size)
    syst_up = 0.001*fakemet
    syst_down = -0.001*fakemet
    met = CandArray.candidatesfromcounts(np.ones_like(counts),
                                         pt=fakemet,
                                         eta=np.zeros_like(counts),
                                         phi=metphi,
                                         mass=np.zeros_like(counts),
                                         MetUnclustEnUpDeltaX=syst_up*np.cos(metphi),
                                         MetUnclustEnUpDeltaY=syst_down*np.sin(metphi))

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

    xform.transform(jets, met=met)

    print('jets',jets.columns)
    print('met',met.columns)

    assert('pt_jer_up' in jets.columns)
    assert('pt_jer_down' in jets.columns)
    assert('mass_jer_up' in jets.columns)
    assert('mass_jer_down' in jets.columns)

    assert('pt_UnclustEn_up' in met.columns)
    assert('pt_UnclustEn_down' in met.columns)
    assert('phi_UnclustEn_up' in met.columns)
    assert('phi_UnclustEn_down' in met.columns)

    for unc in xform.uncertainties:
        assert('pt_'+unc+'_up' in jets.columns)
        assert('pt_'+unc+'_down' in jets.columns)
        assert('mass_'+unc+'_up' in jets.columns)
        assert('mass_'+unc+'_down' in jets.columns)
        assert('pt_'+unc+'_up' in met.columns)
        assert('phi_'+unc+'_up' in met.columns)


def test_corrected_jets_factory():
    import os
    from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack

    events = None
    cache = {}
    from coffea.nanoevents import NanoEventsFactory
    factory = NanoEventsFactory.from_root(os.path.abspath('tests/samples/nano_dy.root'))
    events = factory.events()
    
    jec_stack_names = ['Summer16_23Sep2016V3_MC_L1FastJet_AK4PFPuppi',
                       'Summer16_23Sep2016V3_MC_L2Relative_AK4PFPuppi',
                       'Summer16_23Sep2016V3_MC_L2L3Residual_AK4PFPuppi',
                       'Summer16_23Sep2016V3_MC_L3Absolute_AK4PFPuppi',
                       'Spring16_25nsV10_MC_PtResolution_AK4PFPuppi',
                       'Spring16_25nsV10_MC_SF_AK4PFPuppi']
    for key in evaluator.keys():
        if 'Summer16_23Sep2016V3_MC_UncertaintySources_AK4PFPuppi' in key:
            jec_stack_names.append(key)

    jec_inputs = {name: evaluator[name] for name in jec_stack_names}
    jec_stack = JECStack(jec_inputs)

    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['JetA'] = 'area'
    
    jets = events.Jet
    
    jets['pt_raw'] = jets['rawFactor'] * jets['pt']
    jets['mass_raw'] = jets['rawFactor'] * jets['mass']
    jets['pt_gen'] = awkward1.values_astype(awkward1.fill_none(jets.matched_gen.pt, 0), np.float32)
    print(jets['pt_gen'].layout)
    jets['rho'] = awkward1.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'rho'
    
    events_cache = events.caches[0]
    
    print(name_map)
    
    tic = time.time()
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    toc = time.time()
    
    print('setup corrected jets time =', toc-tic)
    
    tic = time.time()
    #prof = pyinstrument.Profiler()
    #prof.start()
    corrected_jets = jet_factory.build(jets, lazy_cache=events_cache)
    #prof.stop()
    toc = time.time()

    print('corrected_jets build time =', toc-tic)
    
    #sprint(prof.output_text(unicode=True, color=True, show_all=True))
    
    tic = time.time()
    for unc in jet_factory.uncertainties():
        print(unc)
        print(corrected_jets[unc].up.pt)
        print(corrected_jets[unc].down.pt)
    toc = time.time()

    print('build all jet variations =', toc-tic)

    name_map['METpt'] = 'pt'
    name_map['METphi'] = 'phi'
    name_map['METx'] = 'x'
    name_map['METy'] = 'y'
    name_map['JETx'] = 'x'
    name_map['JETy'] = 'y'
    name_map['xMETRaw'] = 'x_raw'
    name_map['yMETRaw'] = 'y_raw'
    name_map['UnClusteredEnergyDeltaX'] = 'MetUnclustEnUpDeltaX'
    name_map['UnClusteredEnergyDeltaY'] = 'MetUnclustEnUpDeltaY'

    tic = time.time()
    met_factory = CorrectedMETFactory(name_map)
    toc = time.time()
    
    print('setup corrected MET time =', toc-tic)


    met = events.MET
    tic = time.time()
    #prof = pyinstrument.Profiler()
    #prof.start()
    corrected_met = met_factory.build(met, corrected_jets, lazy_cache=events_cache)
    #prof.stop()
    toc = time.time()

    #print(prof.output_text(unicode=True, color=True, show_all=True))

    print('corrected_met build time =', toc-tic)

    tic = time.time()
    for unc in (jet_factory.uncertainties() + met_factory.uncertainties()):
        print(unc)
        print(corrected_met[unc].up.pt)
        print(corrected_met[unc].down.pt)
    toc = time.time()
    
    print('build all met variations =', toc-tic)
