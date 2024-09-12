import os

import awkward
import pytest

from coffea.nanoevents import FCC, NanoEventsFactory
from coffea.nanoevents.methods.vector import LorentzVectorRecord


def _events(**kwargs):
    path = os.path.abspath("tests/samples/test_FCC_Spring2021.root")
    factory = NanoEventsFactory.from_root(
        {path: "events"}, schemaclass=FCC.get_schema(version="latest"), **kwargs
    )
    return factory.events()


@pytest.fixture(scope="module")
def eager_events():
    return _events(delayed=False)


@pytest.fixture(scope="module")
def delayed_events():
    return _events(delayed=True)


@pytest.mark.parametrize(
    "field",
    [
        "AllMuonidx0",
        "EFlowNeutralHadron",
        "EFlowNeutralHadron_0",
        "EFlowNeutralHadron_1",
        "EFlowNeutralHadron_2",
        "EFlowNeutralHadronidx0",
        "EFlowNeutralHadronidx1",
        "EFlowNeutralHadronidx2",
        "EFlowPhoton",
        "EFlowPhoton_0",
        "EFlowPhoton_1",
        "EFlowPhoton_2",
        "EFlowPhotonidx0",
        "EFlowPhotonidx1",
        "EFlowPhotonidx2",
        "EFlowTrack",
        "EFlowTrack_0",
        "EFlowTrack_1",
        "EFlowTrackidx0",
        "EFlowTrackidx1",
        "Electronidx0",
        "Jet",
        "Jetidx0",
        "Jetidx1",
        "Jetidx2",
        "Jetidx3",
        "Jetidx4",
        "Jetidx5",
        "MCRecoAssociations",
        "MissingET",
        "MissingETidx0",
        "MissingETidx1",
        "MissingETidx2",
        "MissingETidx3",
        "MissingETidx4",
        "MissingETidx5",
        "Muonidx0",
        "Particle",
        "ParticleIDs",
        "ParticleIDs_0",
        "Particleidx0",
        "Particleidx1",
        "Photonidx0",
        "ReconstructedParticles",
        "ReconstructedParticlesidx0",
        "ReconstructedParticlesidx1",
        "ReconstructedParticlesidx2",
        "ReconstructedParticlesidx3",
        "ReconstructedParticlesidx4",
        "ReconstructedParticlesidx5",
    ],
)
def test_field_is_present(eager_events, delayed_events, field):
    eager_fields = eager_events.fields
    delayed_fields = delayed_events.fields
    assert field in eager_fields
    assert field in delayed_fields


def test_lorentz_behavior(delayed_events):
    assert delayed_events.Particle.behavior["LorentzVector"] == LorentzVectorRecord
    assert (
        delayed_events.ReconstructedParticles.behavior["LorentzVector"]
        == LorentzVectorRecord
    )
    assert isinstance(delayed_events.Particle.eta.compute(), awkward.highlevel.Array)
    assert isinstance(
        delayed_events.ReconstructedParticles.eta.compute(), awkward.highlevel.Array
    )


def test_MC_daughters(delayed_events):
    d = delayed_events.Particle.get_daughters.compute()
    assert isinstance(d, awkward.highlevel.Array)
    assert d.layout.branch_depth[1] == 3


def test_MC_parents(delayed_events):
    p = delayed_events.Particle.get_parents.compute()
    assert isinstance(p, awkward.highlevel.Array)
    assert p.layout.branch_depth[1] == 3


def test_MCRecoAssociations(delayed_events):
    mr = delayed_events.MCRecoAssociations.reco_mc.compute()
    assert isinstance(mr, awkward.highlevel.Array)
    assert mr.layout.branch_depth[1] == 3
