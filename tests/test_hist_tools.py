from __future__ import print_function, division

from coffea import hist
from coffea.util import numpy as np

from dummy_distributions import dummy_jagged_eta_pt
import pytest
import sys

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
    assert h_regular_bins.integrate("x", slice(20, 30)).values()[()][10] == count_some_bin
    assert h_regular_bins.integrate("y", slice(0, 0.3)).values()[()][2] == count_some_bin

    h_reduced = h_regular_bins[10:,-.6:]
    # bin x=1, y=2
    assert h_reduced.integrate("x", slice(20, 30)).values()[()][2] == count_some_bin
    assert h_reduced.integrate("y", slice(0, 0.3)).values()[()][1] == count_some_bin
    h_reduced.fill(x=23, y=0.1)
    assert h_reduced.integrate("x", slice(20, 30)).values()[()][2] == count_some_bin + 1
    assert h_reduced.integrate("y", slice(0, 0.3)).values()[()][1] == count_some_bin + 1

    animal = hist.Cat("animal", "type of animal")
    vocalization = hist.Cat("vocalization", "onomatopoiea is that how you spell it?")
    h_cat_bins = hist.Hist("I like cats", animal, vocalization)
    h_cat_bins.fill(animal="cat", vocalization="meow", weight=2.)
    h_cat_bins.fill(animal="dog", vocalization="meow", weight=np.array([-1., -1., -5.]))
    h_cat_bins.fill(animal="dog", vocalization="woof", weight=100.)
    h_cat_bins.fill(animal="dog", vocalization="ruff")
    assert h_cat_bins.values()[("cat", "meow")] == 2.
    assert h_cat_bins.values(sumw2=True)[("dog", "meow")] == (-7., 27.)
    assert h_cat_bins.integrate("vocalization", ["woof", "ruff"]).values(sumw2=True)[("dog",)] == (101., 10001.)

    height = hist.Bin("height", "height [m]", 10, 0, 5)
    h_mascots_1 = hist.Hist("fermi mascot showdown",
                          animal,
                          vocalization,
                          height,
                          # weight is a reserved keyword
                          hist.Bin("mass", "weight (g=9.81m/s**2) [kg]", np.power(10., np.arange(5)-1)),
                        )

    h_mascots_2 = hist.Hist("fermi mascot showdown",
                          axes=(animal,
                                vocalization,
                                height,
                                # weight is a reserved keyword
                                hist.Bin("mass", "weight (g=9.81m/s**2) [kg]", np.power(10., np.arange(5)-1)),)
                           )

    h_mascots_3 = hist.Hist(
                         axes=[animal,
                               vocalization,
                               height,
                               # weight is a reserved keyword
                               hist.Bin("mass", "weight (g=9.81m/s**2) [kg]", np.power(10., np.arange(5)-1)),],
                            label="fermi mascot showdown"
                          )

    h_mascots_4 = hist.Hist(
                            "fermi mascot showdown",
                            animal,
                            vocalization,
                            height,
                            # weight is a reserved keyword
                            hist.Bin("mass", "weight (g=9.81m/s**2) [kg]", np.power(10., np.arange(5)-1)),
                         axes=[animal,
                               vocalization,
                               height,
                               # weight is a reserved keyword
                               hist.Bin("mass", "weight (g=9.81m/s**2) [kg]", np.power(10., np.arange(5)-1)),],

                       )

    assert h_mascots_1._dense_shape == h_mascots_2._dense_shape
    assert h_mascots_2._dense_shape == h_mascots_3._dense_shape
    assert h_mascots_3._dense_shape == h_mascots_4._dense_shape

    assert h_mascots_1._axes == h_mascots_2._axes
    assert h_mascots_2._axes == h_mascots_3._axes
    assert h_mascots_3._axes == h_mascots_4._axes

    adult_bison_h = np.random.normal(loc=2.5, scale=0.2, size=40)
    adult_bison_w = np.random.normal(loc=700, scale=100, size=40)
    h_mascots_1.fill(animal="bison", vocalization="huff", height=adult_bison_h, mass=adult_bison_w)
    goose_h = np.random.normal(loc=0.4, scale=0.05, size=1000)
    goose_w = np.random.normal(loc=7, scale=1, size=1000)
    h_mascots_1.fill(animal="goose", vocalization="honk", height=goose_h, mass=goose_w)
    crane_h = np.random.normal(loc=1, scale=0.05, size=4)
    crane_w = np.random.normal(loc=10, scale=1, size=4)
    h_mascots_1.fill(animal="crane", vocalization="none", height=crane_h, mass=crane_w)

    with pytest.raises(ValueError):
        h_mascots_1.fill(beast="crane", yelling="none", tallness=crane_h, heavitivity=crane_w)


    h_mascots_2 = h_mascots_1.copy()
    h_mascots_2.clear()
    baby_bison_h = np.random.normal(loc=.5, scale=0.1, size=20)
    baby_bison_w = np.random.normal(loc=200, scale=10, size=20)
    baby_bison_cutefactor = 2.5*np.ones_like(baby_bison_w)
    h_mascots_2.fill(animal="bison", vocalization="baa", height=baby_bison_h, mass=baby_bison_w, weight=baby_bison_cutefactor)
    h_mascots_2.fill(animal="fox", vocalization="none", height=1., mass=30.)

    h_mascots = h_mascots_1 + h_mascots_2
    assert h_mascots.integrate("vocalization", "h*").sum("height", "mass", "animal").values()[()] == 1040.

    species_class = hist.Cat("species_class", "where the subphylum is vertibrates")
    classes = {
        'birds': ['goose', 'crane'],
        'mammals': ['bison', 'fox'],
    }
    h_species = h_mascots.group("animal", species_class, classes)

    assert set(h_species.integrate("vocalization").values().keys()) == set([('birds',), ('mammals',)])
    nbirds_bin = np.sum((goose_h>=0.5)&(goose_h<1)&(goose_w>10)&(goose_w<100))
    nbirds_bin += np.sum((crane_h>=0.5)&(crane_h<1)&(crane_w>10)&(crane_w<100))
    assert h_species.integrate("vocalization").values()[('birds',)][1,2] == nbirds_bin
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
    assert h_species.integrate("vocalization", "h*").axis("height") is height

    tall_class = hist.Cat("tall_class", "species class (species above 1m)")
    mapping = {
        'birds': (['goose', 'crane'], slice(1., None)),
        'mammals': (['bison', 'fox'], slice(1., None)),
    }
    h_tall = h_mascots.group((animal, height), tall_class, mapping)
    tall_bird_count = np.sum(goose_h>=1.) + np.sum(crane_h>=1)
    assert h_tall.sum("mass", "vocalization").values()[('birds',)] == tall_bird_count
    tall_mammal_count = np.sum(adult_bison_h>=1.) + np.sum(baby_bison_h>=1) + 1
    assert h_tall.sum("mass", "vocalization").values()[('mammals',)] == tall_mammal_count

    h_less = h_mascots.remove(["fox", "bison"], axis="animal")
    assert h_less.sum("vocalization", "height", "mass", "animal").values()[()] == 1004.

def test_export1d():
    import uproot
    import os
    from coffea.hist import export1d


    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    h_regular_bins = hist.Hist("regular_joe", hist.Bin("x", "x", 20, 0, 200))
    h_regular_bins.fill(x=test_pt)

    hout = export1d(h_regular_bins)

    filename = 'test_export1d.root'

    with uproot.create(filename) as fout:
        fout['regular_joe'] = hout
        fout.close()

    with uproot.open(filename) as fin:
        hin = fin['regular_joe']


    assert(np.all(hin.edges == hout.edges))
    assert(np.all(hin.values == hout.values))

    del hin
    del fin

    if os.path.exists(filename):
        os.remove(filename)

def test_hist_serdes():
    import pickle
    h_regular_bins = hist.Hist("regular joe",
                               hist.Bin("x", "x", 20, 0, 200),
                               hist.Bin("y", "why", 20, -3, 3))

    h_regular_bins.fill(x=np.array([1.,2.,3.,4.,5.]),y=np.array([-2.,1.,0.,1.,2.]))

    h_regular_bins.sum('x').identifiers('y')

    spkl = pickle.dumps(h_regular_bins)

    hnew = pickle.loads(spkl)

    hnew.sum('x').identifiers('y')

    assert(h_regular_bins._dense_shape == hnew._dense_shape)
    assert(h_regular_bins._axes == hnew._axes)

def test_hist_serdes_labels():
    import pickle
    ax = hist.Bin('asdf', 'asdf', 3, 0, 3)
    ax.identifiers()[0].label = 'type 1'
    h = hist.Hist('a', ax)
    h.identifiers('asdf')

    spkl = pickle.dumps(h)

    hnew = pickle.loads(spkl)

    for old, new in zip(h.identifiers('asdf'), hnew.identifiers('asdf')):
        assert(old.label == new.label)

    assert(h._dense_shape == hnew._dense_shape)
    assert(h._axes == hnew._axes)

@pytest.mark.skipif(sys.version_info < (3, 4), reason="requires python3.4 or higher, test file is pickle proto 4")
def test_hist_compat():
    from coffea.util import load

    test = load('tests/samples/old_hist_format.coffea')

    expected_bins = np.array([ -np.inf, 0.,   20.,   40.,   60.,   80.,  100.,  120.,  140.,
                               160.,  180.,  200.,  220.,  240.,  260.,  280.,  300.,  320.,
                               340.,  360.,  380.,  400.,  420.,  440.,  460.,  480.,  500.,
                               520.,  540.,  560.,  580.,  600.,  620.,  640.,  660.,  680.,
                               700.,  720.,  740.,  760.,  780.,  800.,  820.,  840.,  860.,
                               880.,  900.,  920.,  940.,  960.,  980., 1000., 1020., 1040.,
                              1060., 1080., 1100., 1120., 1140., 1160., 1180., 1200.,   np.inf,
                              np.nan])
    assert np.all(test._axes[2]._interval_bins[:-1] == expected_bins[:-1])
    assert np.isnan(test._axes[2]._interval_bins[-1])

def test_issue_247():
    from coffea import hist

    h = hist.Hist('stuff', hist.Bin('old', 'old', 20, -1, 1))
    h.fill(old=h.axis('old').centers())
    h2 = h.rebin(h.axis('old'), hist.Bin('new', 'new', 10, -1, 1))
    # check first if its even possible to have correct binning
    assert np.all(h2.axis('new').edges() == h.axis('old').edges()[::2])
    # make sure the lookup works properly
    assert np.all(h2.values()[()] == 2.)
    h3 = h.rebin(h.axis('old'), 2)
    assert np.all(h3.values()[()] == 2.)

    with pytest.raises(ValueError):
        # invalid division
        _ = h.rebin(h.axis('old'), hist.Bin('new', 'new', 8, -1, 1))

    newaxis = hist.Bin('new', 'new', h.axis('old').edges()[np.cumsum([0, 2, 3, 5])])
    h4 = h.rebin('old', newaxis)

def test_issue_333():
    axis = hist.Bin("channel", "Channel b1", 50, 0, 2000)
    temp = np.arange(0, 2000, 40, dtype=np.int16)
    assert np.all(axis.index(temp) == np.arange(50) + 1)
