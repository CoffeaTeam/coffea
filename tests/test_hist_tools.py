from __future__ import print_function, division

from coffea import hist
from coffea.util import numpy as np

from dummy_distributions import dummy_jagged_eta_pt

def test_hist():
    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    h_nothing = hist.Hist("empty inside")
    assert h_nothing.sparse_dim() == h_nothing.dense_dim() == 0
    assert h_nothing.values() == {}

    h_regular_bins = hist.Hist("regular joe", hist.Bin("x", "x", 20, 0, 200), hist.Bin("y", "why", 20, -3, 3))
    h_regular_bins.fill(x=test_pt, y=test_eta)
    nentries = np.sum(counts)
    assert h_regular_bins.sum("x", "y", overflow='all').values(sumw2=True)[()] == (nentries, nentries)
    # bin x=2, y=10 (when overflow removed)
    count_some_bin = np.sum((test_pt>=20.)&(test_pt<30.)&(test_eta>=0.)&(test_eta<0.3))
    assert h_regular_bins.project("x", slice(20, 30)).values()[()][10] == count_some_bin
    assert h_regular_bins.project("y", slice(0, 0.3)).values()[()][2] == count_some_bin

    h_reduced = h_regular_bins[10:,-.6:]
    # bin x=1, y=2
    assert h_reduced.project("x", slice(20, 30)).values()[()][2] == count_some_bin
    assert h_reduced.project("y", slice(0, 0.3)).values()[()][1] == count_some_bin
    h_reduced.fill(x=23, y=0.1)
    assert h_reduced.project("x", slice(20, 30)).values()[()][2] == count_some_bin + 1
    assert h_reduced.project("y", slice(0, 0.3)).values()[()][1] == count_some_bin + 1

    animal = hist.Cat("animal", "type of animal")
    vocalization = hist.Cat("vocalization", "onomatopoiea is that how you spell it?")
    h_cat_bins = hist.Hist("I like cats", animal, vocalization)
    h_cat_bins.fill(animal="cat", vocalization="meow", weight=2.)
    h_cat_bins.fill(animal="dog", vocalization="meow", weight=np.array([-1., -1., -5.]))
    h_cat_bins.fill(animal="dog", vocalization="woof", weight=100.)
    h_cat_bins.fill(animal="dog", vocalization="ruff")
    assert h_cat_bins.values()[("cat", "meow")] == 2.
    assert h_cat_bins.values(sumw2=True)[("dog", "meow")] == (-7., 27.)
    assert h_cat_bins.project("vocalization", ["woof", "ruff"]).values(sumw2=True)[("dog",)] == (101., 10001.)

    height = hist.Bin("height", "height [m]", 10, 0, 5)
    h_mascots_1 = hist.Hist("fermi mascot showdown",
                          animal,
                          vocalization,
                          height,
                          # weight is a reserved keyword
                          hist.Bin("mass", "weight (g=9.81m/s**2) [kg]", np.power(10., np.arange(5)-1)),
                        )
    adult_bison_h = np.random.normal(loc=2.5, scale=0.2, size=40)
    adult_bison_w = np.random.normal(loc=700, scale=100, size=40)
    h_mascots_1.fill(animal="bison", vocalization="huff", height=adult_bison_h, mass=adult_bison_w)
    goose_h = np.random.normal(loc=0.4, scale=0.05, size=1000)
    goose_w = np.random.normal(loc=7, scale=1, size=1000)
    h_mascots_1.fill(animal="goose", vocalization="honk", height=goose_h, mass=goose_w)
    crane_h = np.random.normal(loc=1, scale=0.05, size=4)
    crane_w = np.random.normal(loc=10, scale=1, size=4)
    h_mascots_1.fill(animal="crane", vocalization="none", height=crane_h, mass=crane_w)

    h_mascots_2 = h_mascots_1.copy()
    h_mascots_2.clear()
    baby_bison_h = np.random.normal(loc=.5, scale=0.1, size=20)
    baby_bison_w = np.random.normal(loc=200, scale=10, size=20)
    baby_bison_cutefactor = 2.5*np.ones_like(baby_bison_w)
    h_mascots_2.fill(animal="bison", vocalization="baa", height=baby_bison_h, mass=baby_bison_w, weight=baby_bison_cutefactor)
    h_mascots_2.fill(animal="fox", vocalization="none", height=1., mass=30.)

    h_mascots = h_mascots_1 + h_mascots_2
    assert h_mascots.project("vocalization", "h*").sum("height", "mass", "animal").values()[()] == 1040.

    species_class = hist.Cat("species_class", "where the subphylum is vertibrates")
    classes = {
        'birds': ['goose', 'crane'],
        'mammals': ['bison', 'fox'],
    }
    h_species = h_mascots.group(species_class, "animal", classes)

    assert set(h_species.project("vocalization").values().keys()) == set([('birds',), ('mammals',)])
    nbirds_bin = np.sum((goose_h>=0.5)&(goose_h<1)&(goose_w>10)&(goose_w<100))
    nbirds_bin += np.sum((crane_h>=0.5)&(crane_h<1)&(crane_w>10)&(crane_w<100))
    assert h_species.project("vocalization").values()[('birds',)][1,2] == nbirds_bin
    tally = h_species.sum("mass", "height", "vocalization").values()
    assert tally[('birds',)] == 1004.
    assert tally[('mammals',)] == 91.

    h_species.scale({"honk": 0.1, "huff": 0.9}, axis="vocalization")
    h_species.scale(5.)
    tally = h_species.sum("mass", height, vocalization).values(sumw2=True)
    assert tally[('birds',)] == (520., 350.)
    assert tally[('mammals',)] == (435., 25*(40*(0.9**2)+20*(2.5**2)+1))

    assert h_species.axis("vocalization") is vocalization
    assert h_species.axis("height") is height
    assert h_species.project("vocalization", "h*").axis("height") is height

    tall_class = hist.Cat("tall_class", "species class (species above 1m)")
    mapping = {
        'birds': (['goose', 'crane'], slice(1., None)),
        'mammals': (['bison', 'fox'], slice(1., None)),
    }
    h_tall = h_mascots.group(tall_class, (animal, height), mapping)
    tall_bird_count = np.sum(goose_h>=1.) + np.sum(crane_h>=1)
    assert h_tall.sum("mass", "vocalization").values()[('birds',)] == tall_bird_count
    tall_mammal_count = np.sum(adult_bison_h>=1.) + np.sum(baby_bison_h>=1) + 1
    assert h_tall.sum("mass", "vocalization").values()[('mammals',)] == tall_mammal_count
