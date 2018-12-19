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
    
    animal = hist.Cat("animal", "type of animal")
    vocalization = hist.Cat("vocalization", "onomatopoiea is that how you spell it?")
    h_cat_bins = hist.Hist("I like cats", animal, vocalization)
    h_cat_bins.fill(animal="cat", vocalization="meow", weight=2.)
    h_cat_bins.fill(animal="dog", vocalization="meow", weight=np.array([-1., -1., -5.]))
    h_cat_bins.fill(animal="dog", vocalization="woof", weight=100.)
    h_cat_bins.fill(animal="dog", vocalization="ruff")
    assert h_cat_bins.values()[("cat", "meow")] == 2.
    assert h_cat_bins.values(errors=True)[("dog", "meow")] == (-7., np.sqrt(27.))
    assert h_cat_bins.project("vocalization", values=["woof", "ruff"]).values(errors=True)[("dog",)] == (101., np.sqrt(10001.))

    height = hist.Bin("height", "height [m]", 10, 0, 5)
    h_mascots = hist.Hist("fermi mascot showdown",
                          animal,
                          vocalization,
                          height,
                          # weight is a reserved keyword
                          hist.Bin("mass", "weight (g=9.81m/s**2) [kg]", np.power(10., np.arange(5)-1)),
                        )
    adult_bison_h = np.random.normal(loc=2.5, scale=0.2, size=40)
    adult_bison_w = np.random.normal(loc=700, scale=100, size=40)
    h_mascots.fill(animal="bison", vocalization="huff", height=adult_bison_h, mass=adult_bison_w)
    baby_bison_h = np.random.normal(loc=.5, scale=0.1, size=20)
    baby_bison_w = np.random.normal(loc=200, scale=10, size=20)
    baby_bison_cutefactor = 2.5*np.ones_like(baby_bison_w)
    h_mascots.fill(animal="bison", vocalization="baa", height=baby_bison_h, mass=baby_bison_w, weight=baby_bison_cutefactor)
    goose_h = np.random.normal(loc=0.4, scale=0.05, size=1000)
    goose_w = np.random.normal(loc=7, scale=1, size=1000)
    h_mascots.fill(animal="goose", vocalization="honk", height=goose_h, mass=goose_w)
    crane_h = np.random.normal(loc=1, scale=0.05, size=4)
    crane_w = np.random.normal(loc=10, scale=1, size=4)
    h_mascots.fill(animal="crane", vocalization="none", height=crane_h, mass=crane_w)
    h_mascots.fill(animal="fox", vocalization="none", height=1., mass=30.)

    assert h_mascots.project("vocalization", values="h*").project("height").project("mass").project("animal").values()[()] == 1040.

    subphylums = {
        'birds': ['goose', 'crane', 'robin'],
        'mammals': ['bison', 'fox', 'human'],
    }
    h_subphylum = h_mascots.rebin_sparse("animal", new_name="subphylum", new_title="subphylum", mapping=subphylums)

    assert h_subphylum.project("vocalization").values().keys() == [('birds',), ('mammals',)]
    nbirds_bin = np.sum((goose_h>=0.5)&(goose_h<1)&(goose_w>10)&(goose_w<100))
    nbirds_bin += np.sum((crane_h>=0.5)&(crane_h<1)&(crane_w>10)&(crane_w<100))
    assert h_subphylum.project("vocalization").values()[('birds',)][1,2] == nbirds_bin
    tally = h_subphylum.project("mass").project("height").project("vocalization").values()
    assert tally[('birds',)] > tally[('mammals',)]

    assert h_subphylum.axis("vocalization") is vocalization
    assert h_subphylum.axis("height") is height
    assert h_subphylum.project("vocalization", values="h*").axis("height") is height

    # TODO: test clear, add, scale methods
