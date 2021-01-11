from __future__ import print_function, division

from coffea.util import numpy as np
from dummy_distributions import dummy_jagged_eta_pt

import sys
import os.path as osp
import pytest

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
    entrystart = 0
    entrystop = 100
    
    df = LazyDataFrame(tree, entrystart, entrystop, preload_items = ['nMuon'])

    assert(len(df) == 1)
    
    pt = df['Muon_pt']
    assert(len(df) == 2)
    df['Muon_pt_up'] = pt * 1.05
    assert(len(df) == 3)
    assert('Muon_pt' in df.materialized)
    
    assert('Muon_eta' in df.available)
    
    assert(df.size == tree.num_entries)

    with pytest.raises(KeyError):
        x = df['notthere']


@pytest.mark.skipif(sys.platform.startswith("win"), reason='problems with paths on windows')
def test_lazy_dataframe_getattr():
    import uproot
    from coffea.processor import LazyDataFrame
    
    tree = uproot.open(osp.abspath('tests/samples/nano_dy.root'))['Events']
    entrystart = 0
    entrystop = 100
    
    df = LazyDataFrame(tree, entrystart, entrystop, preload_items = ['nMuon'])

    assert(len(df) == 1)
    
    pt = df.Muon_pt
    assert(len(df) == 2)
    assert('Muon_pt' in df.materialized)
    
    assert('Muon_eta' in df.available)
    
    assert(df.size == tree.num_entries)

    with pytest.raises(AttributeError):
        x = df.notthere

    import copy
    df2 = copy.copy(df)
    pt = df2.Muon_pt
    with pytest.raises(AttributeError):
        df2.notthere
