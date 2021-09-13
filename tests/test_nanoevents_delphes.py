import os
import pytest
import numpy as np
from coffea.nanoevents import NanoEventsFactory, DelphesSchema


@pytest.fixture(scope="module")
def events():
    path = os.path.abspath("tests/samples/delphes.root")
    factory = NanoEventsFactory.from_root(
        path, treepath="Delphes", schemaclass=DelphesSchema
    )
    return factory.events()


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
