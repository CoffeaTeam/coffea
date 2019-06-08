from __future__ import print_function, division

from coffea.util import numpy as np
from dummy_distributions import dummy_jagged_eta_pt

import sys
import os.path as osp
import pytest

def test_weights():
    from coffea.processor import Weights
    
    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    scale_central = np.random.normal(loc=1.0, scale=0.01, size=counts.size)
    scale_up = scale_central * 1.10
    scale_down = scale_central * 0.95
    scale_up_shift = 0.10 * scale_central
    scale_down_shift = 0.05 * scale_central

    weight = Weights(counts.size)
    weight.add('test', scale_central, weightUp=scale_up, weightDown=scale_down)
    weight.add('testShift', scale_central, weightUp=scale_up_shift,
               weightDown=scale_down_shift, shift=True)

    var_names = weight.variations
    expected_names = ['testShiftUp', 'testShiftDown', 'testUp', 'testDown']
    for name in expected_names:
        assert(name in var_names)

    test_central = weight.weight()
    exp_weight = scale_central * scale_central

    assert(np.all(np.abs(test_central - (exp_weight)) < 1e-6))

    test_up = weight.weight('testUp')
    exp_up = scale_central * scale_central * 1.10

    assert(np.all(np.abs(test_up - (exp_up)) < 1e-6))

    test_down = weight.weight('testDown')
    exp_down = scale_central * scale_central * 0.95

    assert(np.all(np.abs(test_down - (exp_down)) < 1e-6))

    test_shift_up = weight.weight('testUp')

    assert(np.all(np.abs(test_shift_up - (exp_up)) < 1e-6))
    
    test_shift_down = weight.weight('testDown')

    assert(np.all(np.abs(test_shift_down - (exp_down)) < 1e-6))

def test_packed_selection():
    from coffea.processor import PackedSelection

    sel = PackedSelection()

    counts, test_eta, test_pt = dummy_jagged_eta_pt()

    all_true = np.full(shape=counts.shape, fill_value=True, dtype=np.bool)
    all_false = np.full(shape=counts.shape, fill_value=False, dtype=np.bool)
    ones = np.ones(shape=counts.shape, dtype=np.uint64)
    wrong_shape = ones = np.ones(shape=(counts.shape[0]-5,), dtype=np.bool)

    sel.add('all_true', all_true)
    sel.add('all_false', all_false)

    assert(np.all(sel.require(all_true=True, all_false=False) == all_true))
    assert(np.all(sel.all('all_true', 'all_false') == all_false))

    try:
        sel.require(all_true=1, all_false=0)
    except ValueError:
        pass

    try:
        sel.add('wrong_shape',wrong_shape)
    except ValueError:
        pass

    try:
        sel.add('ones',ones)
    except ValueError:
        pass

    try:
        overpack = PackedSelection()
        for i in range(65):
            overpack.add('sel_%d',all_true)
    except RuntimeError:
        pass

def test_processorabc():
    from coffea.processor import ProcessorABC

    class test(ProcessorABC):
        @property
        def accumulator(self):
            pass
    
        def process(self, df):
            pass
    
        def postprocess(self, accumulator):
            pass
    
    try:
        proc = ProcessorABC()
    except TypeError:
        pass

    proc = test()

    super(test, proc).accumulator

    df = None
    super(test, proc).process(df)

    acc = None
    super(test, proc).postprocess(acc)


@pytest.mark.skipif(sys.platform.startswith("win"), reason='problems with paths on windows')
def test_lazy_dataframe():
    import uproot
    from coffea.processor import LazyDataFrame
    
    tree = uproot.open(osp.abspath('tests/samples/nano_dy.root'))['Events']
    chunksize = 20
    index = 0
    
    df = LazyDataFrame(tree, chunksize, index, preload_items = ['nMuon'])

    assert(len(df) == 1)
    
    pt = df['Muon_pt']
    df['Muon_pt_up'] = pt * 1.05
    assert(len(df) == 3)
    assert('Muon_pt' in df.materialized)
    
    assert(b'Muon_eta' in df.available)
    
    assert(df.size == tree.numentries)

    try:
        x = df['notthere']
    except KeyError:
        pass


@pytest.mark.skipif(sys.platform.startswith("win"), reason='problems with paths on windows')
def test_preloaded_dataframe():
    import uproot
    from coffea.processor import PreloadedDataFrame

    tree = uproot.open(osp.abspath('tests/samples/nano_dy.root'))['Events']
    chunksize = 20
    index = 0
    
    arrays = tree.arrays()
    
    df = PreloadedDataFrame(arrays[b'nMuon'].size, arrays)

    assert(len(arrays) == len(df))
    
    df['nMuon'] = arrays[b'nMuon']
    assert('nMuon' in df.available)
    
    assert(np.all(df[b'nMuon'] == arrays[b'nMuon']))

    assert(b'Muon_eta' in df.available)
    assert(b'nMuon' in df.materialized)
    assert(df.size == arrays[b'nMuon'].size)
