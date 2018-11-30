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

