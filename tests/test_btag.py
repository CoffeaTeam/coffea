import pytest
from coffea.btag_tools import BTagScaleFactor
from dummy_distributions import dummy_jagged_eta_pt
import numpy

def test_BTagScalefactor():
    sf1 = BTagScaleFactor('tests/samples/testBTagSF.btag.csv', 'medium')
    sf2 = BTagScaleFactor('tests/samples/DeepCSV_102XSF_V1.btag.csv.gz', BTagScaleFactor.RESHAPE, 'iterativefit')
    sf3 = BTagScaleFactor('tests/samples/DeepCSV_102XSF_V1.btag.csv.gz', BTagScaleFactor.TIGHT)
    sf4 = BTagScaleFactor('tests/samples/DeepCSV_2016LegacySF_V1_TuneCP5.btag.csv.gz', BTagScaleFactor.RESHAPE, 'iterativefit', keep_df=True)
    # import pdb; pdb.set_trace()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    test_flavor = numpy.random.choice([0, 4, 5], size=len(test_eta))
    test_allb = numpy.ones_like(test_flavor) * 5
    test_discr = numpy.random.rand(len(test_eta))

    sf1.eval('central', test_flavor, test_eta, test_pt, test_discr)
    sf1.eval('up', test_flavor, test_eta, test_pt)
    sf2.eval('central', test_allb, test_eta, test_pt, test_discr)
    with pytest.raises(ValueError):
        sf2.eval('up', test_allb, test_eta, test_pt)
    sf3.eval('central', test_flavor, test_eta, test_pt, test_discr)
    sf3.eval('up', test_flavor, test_eta, test_pt)
    with pytest.raises(ValueError):
        sf4.eval('central', test_flavor, test_eta, test_pt, test_discr)
    sf4.eval('central', test_allb, test_eta, test_pt, test_discr)
