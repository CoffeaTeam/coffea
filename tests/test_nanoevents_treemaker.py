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
    [
        "HT"
        "MET",
        "Muons",
        "Electrons",
        "Photons",
        "Jets",
        "JetsAK8",
        "Tracks",
        "PrimaryVertices",
        "GenParticles"
    ],
)
def test_collection_exists(events, collection):
    assert hasattr(events, collection)


@pytest.mark.parametrize(
    "collection",
    [
        "Muons",
        "Electrons",
        "Photons",
        "Jets",
        "JetsAK8",
        "Tracks",
        "GenParticles"
    ],
)
def test_lorentz_vectorization(collection, events):
    mask = ak.num(events[collection]) > 0
    assert (
        ak.type(events[collection][mask][0, 0]).parameters["__record__"]
        == "LorentzVector"
    )


@pytest.mark.parametrize(
    "collection",
    [
        "CaloJet02",
        "CaloJet04",
        "CaloJet08",
        "CaloJet15",
        "GenJet",
        "GenJet02",
        "GenJet04",
        "GenJet08",
        "GenJet15",
        "Jet",
        "ParticleFlowJet02",
        "ParticleFlowJet04",
        "ParticleFlowJet08",
        "ParticleFlowJet15",
        "TrackJet02",
        "TrackJet04",
        "TrackJet08",
        "TrackJet15",
    ],
)
def test_nested_lorentz_vectorization(collection, events):
    mask = ak.num(events[collection]) > 0
    assert ak.all(ak.num(events[collection].PrunedP4_5, axis=2) == 5)
    assert (
        ak.type(events[collection][mask].PrunedP4_5[0, 0, 0]).parameters["__record__"]
        == "LorentzVector"
    )

    assert ak.all(ak.num(events[collection].SoftDroppedP4_5, axis=2) == 5)
    assert (
        ak.type(events[collection][mask].SoftDroppedP4_5[0, 0, 0]).parameters[
            "__record__"
        ]
        == "LorentzVector"
    )

    assert ak.all(ak.num(events[collection].TrimmedP4_5, axis=2) == 5)
    assert (
        ak.type(events[collection][mask].TrimmedP4_5[0, 0, 0]).parameters["__record__"]
        == "LorentzVector"
    )
