import os
from coffea.nanoaod import NanoEvents


def genroundtrips(genpart):
    # check genpart roundtrip
    assert (genpart.children.parent.pdgId == genpart.pdgId).all().all().all()
    assert (genpart.parent.children.pdgId == genpart.pdgId).any().fillna(True).all().all()
    # distinctParent should be distinct and it should have a relevant child
    assert (genpart.distinctParent.pdgId != genpart.pdgId).fillna(True).all().all()
    assert (genpart.distinctParent.children.pdgId == genpart.pdgId).any().fillna(True).all().all()
    # exercise hasFlags
    genpart.hasFlags(['isHardProcess'])
    genpart.hasFlags(['isHardProcess', 'isDecayedLeptonHadron'])


def crossref(events):
    # check some cross-ref roundtrips (some may not be true always but they are for the test file)
    assert (events.Jet.matched_muons.matched_jet.pt == events.Jet.pt).all().all().all()
    assert (events.Electron.matched_photon.matched_electron.r9 == events.Electron.r9).all().all()


def test_read_nanomc():
    events = NanoEvents.from_file(os.path.abspath('tests/samples/nano_dy.root'))

    # test after views first
    genroundtrips(events.GenPart[events.GenPart.eta > 0])
    genroundtrips(events[(events.Electron.eta > 0).any()].GenPart)
    genroundtrips(events.GenPart)

    # sane gen matching (note for electrons gen match may be photon(22))
    assert ((abs(events.Electron.matched_gen.pdgId) == 11) | (events.Electron.matched_gen.pdgId == 22)).all().all()
    assert (abs(events.Muon.matched_gen.pdgId) == 13).all().all()

    genroundtrips(events.Electron.matched_gen)

    crossref(events[events.Jet.counts > 2])
    crossref(events)

    assert events.Photon.isTight.any().tolist()[:9] == [False, True, True, True, False, False, False, False, False]


def test_read_nanodata():
    events = NanoEvents.from_file(os.path.abspath('tests/samples/nano_dimuon.root'))

    crossref(events)
    crossref(events[events.Jet.counts > 2])
