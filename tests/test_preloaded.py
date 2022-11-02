import pytest
import os
import uproot
import awkward
from coffea.processor.test_items import NanoEventsProcessor
from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.mapping import SimplePreloadedColumnSource


def test_preloaded_nanoevents():
    columns = [
        "nMuon",
        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_mass",
        "Muon_charge",
        "nJet",
        "Jet_eta",
    ]
    p = NanoEventsProcessor(columns=columns)

    rootdir = uproot.open(os.path.abspath("tests/samples/nano_dy.root"))
    tree = rootdir["Events"]
    arrays = tree.arrays(columns, how=dict)
    src = SimplePreloadedColumnSource(
        arrays, rootdir.file.uuid, tree.num_entries, object_path="/Events"
    )
    print(arrays, tree.num_entries)

    print(src.metadata)

    print(awkward.to_buffers(tree.arrays(columns)))

    for k, v in arrays.items():
        print(k, len(v))

    # just make the from_buffers ourselves for now
    form = {
        "class": "ListOffsetArray",
        "offsets": "i64",
        "content": {
            "class": "RecordArray",
            "fields": ["pt", "eta", "phi", "mass", "charge"],
            "contents": [
                {
                    "class": "NumpyArray",
                    "primitive": "float32",
                    "parameters": {"__doc__": "Muon_pt"},
                    "form_key": "Muon_pt%2C%21load%2C%21content",
                },
                {
                    "class": "NumpyArray",
                    "primitive": "float32",
                    "parameters": {"__doc__": "Muon_eta"},
                    "form_key": "Muon_eta%2C%21load%2C%21content",
                },
                {
                    "class": "NumpyArray",
                    "primitive": "float32",
                    "parameters": {"__doc__": "Muon_phi"},
                    "form_key": "Muon_phi%2C%21load%2C%21content",
                },
                {
                    "class": "NumpyArray",
                    "primitive": "float32",
                    "parameters": {"__doc__": "Muon_mass"},
                    "form_key": "Muon_mass%2C%21load%2C%21content",
                },
                {
                    "class": "NumpyArray",
                    "primitive": "int32",
                    "parameters": {"__doc__": "Muon_charge"},
                    "form_key": "Muon_charge%2C%21load%2C%21content",
                },
            ],
            "parameters": {
                "__record__": "Muon",
                "__doc__": "nMuon",
                "collection_name": "Muon",
            },
            "form_key": "%21invalid%2CMuon",
        },
        "form_key": "nMuon%2C%21load%2C%21counts2offsets%2C%21skip",
    }

    muon_array = awkward.from_buffers(
        form=form,
        length=tree.num_entries,
        container=arrays,
    )

    print(muon_array)

    events = NanoEventsFactory.from_preloaded(
        src, metadata={"dataset": "ZJets"}
    ).events()
    hists = p.process(events)

    print(hists)
    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6

    with pytest.raises(AttributeError):
        print(events.Muon.matched_jet)
