from __future__ import print_function, division

from fnal_column_analysis_tools import lookup_tools
import uproot
import awkward
import numpy as np

from dummy_distributions import dummy_pt_eta

def test_lookup_tools():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(["testSF2d scalefactors_Tight_Electron tests/samples/testSF2d.root"])
    extractor.finalize()

    evaluator = extractor.make_evaluator()

    counts, test_in1, test_in2 = dummy_pt_eta()

    # test flat eval
    test_out = evaluator["testSF2d"](test_in1, test_in2)

    # test structured eval
    test_in1_jagged = awkward.JaggedArray.fromcounts(counts, test_in1)
    test_in2_jagged = awkward.JaggedArray.fromcounts(counts, test_in2)
    test_out_jagged = evaluator["testSF2d"](test_in1_jagged, test_in2_jagged)

    assert (test_out_jagged.counts==counts).all()
    assert (test_out==test_out_jagged.flatten()).all()

    # From make_expected_lookup.py
    expected_output = np.array([
       0.        , 0.98826289, 0.        , 1.02135682, 1.01189065,
       0.        , 1.05108559, 0.98096448, 0.        , 0.98948598,
       0.95345747, 0.97203946, 0.90893763, 0.97466666, 0.98948598,
       1.01189065, 0.88245934, 0.96855348, 0.        , 0.        ,
       0.90893763, 0.96698761, 0.        , 0.95782316, 0.97893435,
       0.97893435, 0.95782316, 0.        , 0.98948598, 0.88245934,
       0.93766236, 0.97981155, 0.97466666, 0.        , 0.97167486,
       0.97167486, 0.97520661, 0.97463286, 1.02135682, 0.9529984 ,
       0.88245934, 0.97893435, 1.00716329, 0.98199672, 0.96855348,
       0.97549593, 0.98199672, 0.95308644, 0.        , 0.94039732,
       0.        , 0.97893435, 0.97203946, 1.01189065, 0.90780139,
       1.00759494, 0.9529984 , 0.97990727, 0.98096448, 0.        ,
       0.97167486, 0.88245934])

    diff = np.abs(test_out-expected_output)
    print("Max diff: %.16f" % diff.max())
    print("Median diff: %.16f" % np.median(diff))
    print("Diff over threshold rate: %.1f %%" % (100*(diff >= 1.e-8).sum()/diff.size))
    assert (diff < 1.e-8).all()

