from __future__ import print_function, division

from coffea import lookup_tools
import uproot
from coffea.util import awkward
from coffea.util import numpy as np

from dummy_distributions import dummy_jagged_eta_pt


def test_extractor_exceptions():
    extractor = lookup_tools.extractor()

    # test comments
    extractor.add_weight_sets(["#testSF2d asdfgh tests/samples/testSF2d.histo.root"])
    
    # test malformed statement
    try:
        extractor.add_weight_sets(["testSF2d testSF2d asdfgh tests/samples/testSF2d.histo.root"])
    except Exception as e:
        assert(e.args[0] == '"testSF2d testSF2d asdfgh tests/samples/testSF2d.histo.root" not formatted as "<local name> <name> <weights file>"')
    
    # test not existant file entry
    try:
         extractor.add_weight_sets(["testSF2d asdfgh tests/samples/testSF2d.histo.root"])
    except Exception as e:
        assert(e.args[0] == 'Weights named "asdfgh" not in tests/samples/testSF2d.histo.root!')

    # test unfinalized evaluator
    try:
        extractor.add_weight_sets(["testSF2d scalefactors_Tight_Electron tests/samples/testSF2d.histo.root"])
        extractor.make_evaluator()
    except Exception as e:
        assert(e.args[0] == 'Cannot make an evaluator from unfinalized extractor!')

def test_evaluator_exceptions():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(["testSF2d scalefactors_Tight_Electron tests/samples/testSF2d.histo.root"])

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    test_eta_jagged = awkward.JaggedArray.fromcounts(counts, test_eta)
    test_pt_jagged = awkward.JaggedArray.fromcounts(counts, test_pt)

    extractor.finalize()
    evaluator = extractor.make_evaluator()
    
    try:
        test_out = evaluator["testSF2d"](test_pt_jagged, test_eta)
    except Exception as e:
        assert(e.args[0] == 'do not mix JaggedArrays and numpy arrays when calling a derived class of lookup_base')

def test_evaluator_exceptions():
    from coffea.lookup_tools.lookup_base import lookup_base
    try:
        lookup_base()._evaluate()
    except NotImplementedError:
        pass

def test_root_scalefactors():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(["testSF2d scalefactors_Tight_Electron tests/samples/testSF2d.histo.root"])
    
    extractor.finalize(reduce_list=['testSF2d'])

    evaluator = extractor.make_evaluator()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    
    # test flat eval
    test_out = evaluator["testSF2d"](test_eta, test_pt)

    # print it
    print(evaluator["testSF2d"])
    
    # test structured eval
    test_eta_jagged = awkward.JaggedArray.fromcounts(counts, test_eta)
    test_pt_jagged = awkward.JaggedArray.fromcounts(counts, test_pt)
    test_out_jagged = evaluator["testSF2d"](test_eta_jagged, test_pt_jagged)

    assert (test_out_jagged.counts==counts).all()
    assert (test_out==test_out_jagged.flatten()).all()

    # From make_expected_lookup.py
    expected_output = np.array([
       0.90780139, 0.82748538, 0.86332178, 0.86332178, 0.97981155,
       0.79701495, 0.88245934, 0.82857144, 0.91884059, 0.97466666,
       0.94072163, 1.00775194, 0.82748538, 1.00775194, 0.97203946,
       0.98199672, 0.80655736, 0.90893763, 0.88245934, 0.79701495,
       0.82748538, 0.82857144, 0.91884059, 0.90893763, 0.97520661,
       0.97520661, 0.82748538, 0.91884059, 0.97203946, 0.88245934,
       0.79701495, 0.9458763 , 1.00775194, 0.80655736, 1.00775194,
       1.00775194, 0.98976982, 0.98976982, 0.86332178, 0.94072163,
       0.80655736, 0.98976982, 0.96638656, 0.9458763 , 0.90893763,
       0.9529984 , 0.9458763 , 0.9529984 , 0.80655736, 0.80655736,
       0.80655736, 0.98976982, 0.97466666, 0.98199672, 0.86332178,
       1.03286386, 0.94072163, 1.03398061, 0.82857144, 0.80655736,
       1.00775194, 0.80655736])

    diff = np.abs(test_out-expected_output)
    print("Max diff: %.16f" % diff.max())
    print("Median diff: %.16f" % np.median(diff))
    print("Diff over threshold rate: %.1f %%" % (100*(diff >= 1.e-8).sum()/diff.size))
    assert (diff < 1.e-8).all()


def test_btag_csv_scalefactors():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(["testBTag * tests/samples/testBTagSF.btag.csv"])
    extractor.finalize()

    evaluator = extractor.make_evaluator()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    # discriminant used for reshaping, zero otherwise
    test_discr = np.zeros_like(test_eta)

    print(evaluator['testBTagCSVv2_1_comb_up_0'])
    
    sf_out = evaluator['testBTagCSVv2_1_comb_up_0'](test_eta, test_pt, test_discr)


def test_histo_json_scalefactors():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(["testJson * tests/samples/EIDISO_WH_out.histo.json"])
    extractor.finalize()
    
    evaluator = extractor.make_evaluator()
    
    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    
    sf_out = evaluator['testJsonEIDISO_WH/eta_pt_ratio_value'](test_eta, test_pt)
    sf_err_out = evaluator['testJsonEIDISO_WH/eta_pt_ratio_error'](test_eta, test_pt)
    print(sf_out)
    print(sf_err_out)

def test_jec_txt_scalefactors():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets([
        "testJEC * tests/samples/Fall17_17Nov2017_V32_MC_L2Relative_AK4PFPuppi.jec.txt",
        "* * tests/samples/Summer16_07Aug2017_V11_L1fix_MC_L2Relative_AK4PFchs.jec.txt.gz",
        "* * tests/samples/Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFPuppi.junc.txt",
        "* * tests/samples/Autumn18_V8_MC_UncertaintySources_AK4PFchs.junc.txt",
        "* * tests/samples/Spring16_25nsV10_MC_SF_AK4PFPuppi.jersf.txt"
    ])
    extractor.finalize()

    evaluator = extractor.make_evaluator()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    
    jec_out = evaluator['testJECFall17_17Nov2017_V32_MC_L2Relative_AK4PFPuppi'](test_eta,test_pt)

    print(evaluator['testJECFall17_17Nov2017_V32_MC_L2Relative_AK4PFPuppi'])
    
    jec_out = evaluator['Summer16_07Aug2017_V11_L1fix_MC_L2Relative_AK4PFchs'](test_eta,test_pt)
    
    print(evaluator['Summer16_07Aug2017_V11_L1fix_MC_L2Relative_AK4PFchs'])

    jersf = evaluator['Spring16_25nsV10_MC_SF_AK4PFPuppi']
    
    print(evaluator['Spring16_25nsV10_MC_SF_AK4PFPuppi'])
    
    junc_out = evaluator['Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFPuppi'](test_eta,test_pt)

    print(evaluator['Fall17_17Nov2017_V32_MC_Uncertainty_AK4PFPuppi'])
    
    assert('Autumn18_V8_MC_UncertaintySources_AK4PFchs_AbsoluteScale' in evaluator.keys())
    junc_out = evaluator['Autumn18_V8_MC_UncertaintySources_AK4PFchs_AbsoluteScale'](test_eta,test_pt)
    print(evaluator['Autumn18_V8_MC_UncertaintySources_AK4PFchs_AbsoluteScale'])
