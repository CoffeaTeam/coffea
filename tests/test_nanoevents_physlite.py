import os
from coffea.nanoevents import NanoEventsFactory, PHYSLITESchema


def test_electron_track_links():
    path = os.path.abspath("tests/samples/DAOD_PHYSLITE_21.2.108.0.art.pool.root")
    factory = NanoEventsFactory.from_root(
        path, treepath="CollectionTree", schemaclass=PHYSLITESchema
    )
    events = factory.events()
    for event in events:
        for electron in event.Electrons:
            for link_index, link in enumerate(electron.trackParticleLinks):
                track_index = link.m_persIndex
                assert (
                    event.GSFTrackParticles[track_index].z0
                    == electron.trackParticles[link_index].z0
                )
