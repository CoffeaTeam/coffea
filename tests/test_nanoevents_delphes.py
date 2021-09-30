import os
import pytest
from coffea.nanoevents import NanoEventsFactory, DelphesSchema
import awkward as ak


@pytest.fixture(scope="module")
def events():
    path = os.path.abspath("tests/samples/delphes.root")
    factory = NanoEventsFactory.from_root(
        path, treepath="Delphes", schemaclass=DelphesSchema
    )
    return factory.events()


def test_listify(events):
    assert ak.to_list(events[0])


@pytest.mark.parametrize(
    "collection",
    [
        "CaloJet02",
        "CaloJet04",
        "CaloJet08",
        "CaloJet15",
        "EFlowNeutralHadron",
        "EFlowPhoton",
        "EFlowTrack",
        "Electron",
        "Event",
        "EventLHEF",
        "GenJet",
        "GenJet02",
        "GenJet04",
        "GenJet08",
        "GenJet15",
        "GenMissingET",
        "Jet",
        "MissingET",
        "Muon",
        "Particle",
        "ParticleFlowJet02",
        "ParticleFlowJet04",
        "ParticleFlowJet08",
        "ParticleFlowJet15",
        "Photon",
        "ScalarHT",
        "Tower",
        "Track",
        "TrackJet02",
        "TrackJet04",
        "TrackJet08",
        "TrackJet15",
        "WeightLHEF",
    ],
)
def test_collection_exists(events, collection):
    assert hasattr(events, collection)


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
def test_lorentz_vectorization(collection, events):
    mask = ak.num(events[collection]) > 0
    assert (
        ak.type(events[collection][mask][0, 0].Area).parameters["__record__"]
        == "LorentzVector"
    )
    assert (
        ak.type(events[collection][mask][0, 0].SoftDroppedJet).parameters["__record__"]
        == "LorentzVector"
    )
    assert (
        ak.type(events[collection][mask][0, 0].SoftDroppedSubJet1).parameters[
            "__record__"
        ]
        == "LorentzVector"
    )
    assert (
        ak.type(events[collection][mask][0, 0].SoftDroppedSubJet2).parameters[
            "__record__"
        ]
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
