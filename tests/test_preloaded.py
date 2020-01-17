import pytest
import os
import uproot
from coffea import processor
from coffea.processor.test_items import NanoEventsProcessor
from coffea.nanoaod import NanoEvents


def test_preloaded_nanoevents():
    columns = ['nMuon','Muon_pt','Muon_eta','Muon_phi','Muon_mass','Muon_charge', 'nJet', 'Jet_eta']
    p = NanoEventsProcessor(columns=columns)

    tree = uproot.open(os.path.abspath('tests/samples/nano_dy.root'))['Events']
    arrays = tree.arrays(columns, flatten=True, namedecode='ascii')
    df = processor.PreloadedDataFrame(tree.numentries, arrays)
    print(arrays)

    events = NanoEvents.from_arrays(arrays, metadata={'dataset': 'ZJets'})
    hists = p.process(events)

    print(hists)
    assert( hists['cutflow']['ZJets_pt'] == 18 )
    assert( hists['cutflow']['ZJets_mass'] == 6 )

    with pytest.raises(RuntimeError):
        print(events.Muon.matched_jet)
