from fnal_column_analysis_tools import lookup_tools
import uproot
import awkward
import numpy as np


def test_lookup_tools():
    extractor = lookup_tools.extractor()
    extractor.add_weight_sets(["testSF2d scalefactors_Tight_Electron tests/samples/testSF2d.root"])
    extractor.finalize()

    evaluator = extractor.make_evaluator()

    # dummy distributions
    np.random.seed(42)
    counts = np.random.exponential(2, size=20).astype(int)
    entries = np.sum(counts)
    test_in1 = np.random.exponential(50., size=entries)
    test_in2 = np.random.uniform(-3., 3., size=entries)

    # test flat eval
    test_out = evaluator["testSF2d"](test_in1, test_in2)

    # test structured eval
    test_in1_jagged = awkward.JaggedArray.fromcounts(counts, test_in1)
    test_in2_jagged = awkward.JaggedArray.fromcounts(counts, test_in2)
    test_out_jagged = evaluator["testSF2d"](test_in1_jagged, test_in2_jagged)

    assert (test_out_jagged.counts==counts).all()
    assert (test_out==test_out_jagged.flatten()).all()

    # TODO: determine expected output

