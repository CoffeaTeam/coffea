import os

import numpy as np
import pytest

from coffea.nanoevents import NanoEventsFactory, PHYSLITESchema

pytestmark = pytest.mark.skip(reason="uproot is upset with this file...")


def _events():
    path = os.path.abspath("tests/samples/DAOD_PHYSLITE_21.2.108.0.art.pool.root")
    factory = NanoEventsFactory.from_root(
        path,
        treepath="CollectionTree",
        schemaclass=PHYSLITESchema,
        permit_dask=False,
    )
    return factory.events()


@pytest.fixture(scope="module")
def events():
    return _events()


@pytest.mark.parametrize("do_slice", [False, True])
def test_electron_track_links(events, do_slice):
    if do_slice:
        events = events[np.random.randint(2, size=len(events)).astype(bool)]
    for event in events:
        for electron in event.Electrons:
            for link_index, link in enumerate(electron.trackParticleLinks):
                track_index = link.m_persIndex
                print(track_index)
                print(event.GSFTrackParticles)
                print(electron.trackParticleLinks)
                print(electron.trackParticles)

                assert (
                    event.GSFTrackParticles[track_index].z0
                    == electron.trackParticles[link_index].z0
                )


# from MetaData/EventFormat
_hash_to_target_name = {
    13267281: "TruthPhotons",
    342174277: "TruthMuons",
    368360608: "TruthNeutrinos",
    375408000: "TruthTaus",
    394100163: "TruthElectrons",
    614719239: "TruthBoson",
    660928181: "TruthTop",
    779635413: "TruthBottom",
}


def test_truth_links_toplevel(events):
    children_px = events.TruthBoson.children.px
    for i_event, event in enumerate(events):
        for i_particle, particle in enumerate(event.TruthBoson):
            for i_link, link in enumerate(particle.childLinks):
                assert (
                    event[_hash_to_target_name[link.m_persKey]][link.m_persIndex].px
                    == children_px[i_event][i_particle][i_link]
                )


def test_truth_links(events):
    for i_event, event in enumerate(events):
        for i_particle, particle in enumerate(event.TruthBoson):
            for i_link, link in enumerate(particle.childLinks):
                assert (
                    event[_hash_to_target_name[link.m_persKey]][link.m_persIndex].px
                    == particle.children[i_link].px
                )
