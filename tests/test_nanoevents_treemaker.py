import os
import pytest
from coffea.nanoevents import NanoEventsFactory, TreeMakerSchema
import awkward as ak


@pytest.fixture(scope="module")
def events():
    path = os.path.abspath("tests/samples/treemaker.root")
    factory = NanoEventsFactory.from_root(
        path, treepath="PreSelection", schemaclass=TreeMakerSchema
    )
    return factory.events()


def test_listify(events):
    assert ak.to_list(events[0])


@pytest.mark.parametrize(
    "collection",
    ["HT", "MET", "Weight"],
)
def test_collection_exists(events, collection):
    assert hasattr(events, collection)


@pytest.mark.parametrize(
    "collection,arr_type",
    [
        ("Muons", "PtEtaPhiELorentzVector"),
        (
            "Electrons",
            "PtEtaPhiELorentzVector",
        ),
        ("Photons", "PtEtaPhiELorentzVector"),
        ("Jets", "PtEtaPhiELorentzVector"),
        ("JetsAK8", "PtEtaPhiELorentzVector"),
        ("Tracks", "LorentzVector"),
        ("GenParticles", "PtEtaPhiELorentzVector"),
        ("PrimaryVertices", "ThreeVector"),
    ],
)
def test_lorentzvector_behavior(collection, arr_type, events):
    assert ak.type(events[collection])
    assert ak.type(events[collection]).type.type.__str__().startswith(arr_type)


@pytest.mark.parametrize(
    "collection,subcollection,arr_type,element",
    [
        ("JetsAK8", "subjets", "PtEtaPhiELorentzVector", "pt"),
        ("Tracks", "hitPattern", "int32", None),
    ],
)
def test_nested_collection(collection, subcollection, arr_type, element, events):
    assert ak.type(events[collection][subcollection])
    assert ak.type(events[collection][subcollection + "Counts"])
    assert (
        ak.type(events[collection][subcollection])
        .type.type.type.__str__()
        .startswith(arr_type)
    )
    if element is None:
        assert ak.all(
            events[collection][subcollection + "Counts"]
            == ak.count(events[collection][subcollection], axis=-1)
        )
    else:
        assert ak.all(
            events[collection][subcollection + "Counts"]
            == ak.count(events[collection][subcollection][element], axis=-1)
        )
