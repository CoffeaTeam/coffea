from __future__ import print_function, division

from fnal_column_analysis_tools import hist
import numpy as np

from dummy_distributions import dummy_jagged_eta_pt

def test_hist():
    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    h_nothing = hist.Hist("empty inside")
    assert h_nothing.sparse_dim() == h_nothing.dense_dim() == 0
    assert h_nothing.values() == {}

    h_regular_bins = hist.Hist("regular joe", hist.Bin("x", "x", 20, 0, 200), hist.Bin("y", "why", 50, -3, 3))
    h_regular_bins.fill(x=test_pt, y=test_eta)
    nentries = np.sum(counts)
    assert h_regular_bins.project("x").project("y").values(errors=True)[()] == (nentries, np.sqrt(nentries))
    count_some_bin = np.sum((test_pt>=0.)&(test_pt<10.)&(test_eta>=0.)&(test_eta<0.12))
    assert h_regular_bins.project("y", lo_hi=(0, 0.12)).values()[()][0] == count_some_bin

    h_cat_bins = hist.Hist("I like cats", hist.Cat("animal", "type of animal"), hist.Cat("vocalization", "onomatopoiea is that how you spell it?"))
    h_cat_bins.fill(animal="cat", vocalization="meow", weight=2.)
    h_cat_bins.fill(animal="dog", vocalization="meow", weight=np.array([-1., -1., -5.]))
    h_cat_bins.fill(animal="dog", vocalization="woof", weight=100.)
    h_cat_bins.fill(animal="dog", vocalization="ruff")
    assert h_cat_bins.values()[("cat", "meow")] == 2.
    print(h_cat_bins._sumw2)
    assert h_cat_bins.project("vocalization", values=["woof", "ruff"]).values(errors=True)[("dog",)] == (101., np.sqrt(10001.))
