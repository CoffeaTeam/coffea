import os

import dask
import pytest

from coffea.nanoevents import NanoEventsFactory, PHYSLITESchema


def _events():
    path = os.path.abspath("tests/samples/DAOD_PHYSLITE_21.2.108.0.art.pool.root")
    factory = NanoEventsFactory.from_root(
        {path: "CollectionTree"},
        schemaclass=PHYSLITESchema,
        delayed=True,
    )
    return factory.events()


@pytest.fixture(scope="module")
def events():
    return _events()


def test_load_single_field_of_linked(events):
    with dask.config.set({"awkward.raise-failed-meta": True}):
        events.Electrons.caloClusters.calE.compute()


@pytest.mark.parametrize("do_slice", [False, True])
def test_electron_track_links(events, do_slice):
    if do_slice:
        events = events[::2]
    trackParticles = events.Electrons.trackParticles.compute()
    for i, event in enumerate(events[["Electrons", "GSFTrackParticles"]].compute()):
        for j, electron in enumerate(event.Electrons):
            for link_index, link in enumerate(electron.trackParticleLinks):
                track_index = link.m_persIndex
                assert (
                    event.GSFTrackParticles[track_index].z0
                    == trackParticles[i][j][link_index].z0
                )
