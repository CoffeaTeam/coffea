from __future__ import print_function, division

from fnal_column_analysis_tools import hist
import numpy as np

from dummy_distributions import dummy_jagged_eta_pt

def test_hist():
    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    h_regular_bins = hist.Hist(Bin("x", "x", 20, 0, 200), Bin("y", "why", 50, -3, 3))
    h_regular_bins.fill(x=test_pt, y=test_eta)

    nentries = np.sum(counts)
    assert h_regular_bins.project("x").project("y").values(errors=True)[()] == (nentries, np.sqrt(nentries))

    count_some_bin = np.sum((test_pt>=0.)&(test_pt<10.)&(test_eta>=0.)&(test_eta<0.12))
    assert h_regular_bins.project("y", lo_hi=(0, 0.12)).values()[()][0] == count_some_bin

