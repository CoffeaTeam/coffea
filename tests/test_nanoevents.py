import os
import awkward1 as ak
from coffea.nanoevents import NanoEventsFactory


def genroundtrips(genpart):
    # check genpart roundtrip
    assert ak.all(genpart.children.parent.pdgId == genpart.pdgId)
    assert ak.all(ak.any(genpart.parent.children.pdgId == genpart.pdgId, axis=-1, mask_identity=True))
    # distinctParent should be distinct and it should have a relevant child
    assert ak.all(genpart.distinctParent.pdgId != genpart.pdgId)
    assert ak.all(ak.any(genpart.distinctParent.children.pdgId == genpart.pdgId, axis=-1, mask_identity=True))
    # exercise hasFlags
    genpart.hasFlags(['isHardProcess'])
    genpart.hasFlags(['isHardProcess', 'isDecayedLeptonHadron'])


def crossref(events):
    # check some cross-ref roundtrips (some may not be true always but they are for the test file)
    assert ak.all(events.Jet.matched_muons.matched_jet.pt == events.Jet.pt)
    assert ak.all(events.Electron.matched_photon.matched_electron.r9 == events.Electron.r9)


def test_read_nanomc():
    factory = NanoEventsFactory.from_file(os.path.abspath('tests/samples/nano_dy.root'))
    events = factory.events()

    # test after views first
    genroundtrips(events.GenPart.mask[events.GenPart.eta > 0])
    genroundtrips(events.mask[ak.any(events.Electron.pt > 50, axis=1)].GenPart)
    genroundtrips(events.GenPart)

    genroundtrips(events.GenPart[events.GenPart.eta > 0])
    genroundtrips(events[ak.any(events.Electron.pt > 50, axis=1)].GenPart)

    # sane gen matching (note for electrons gen match may be photon(22))
    assert ak.all((abs(events.Electron.matched_gen.pdgId) == 11) | (events.Electron.matched_gen.pdgId == 22))
    assert ak.all(abs(events.Muon.matched_gen.pdgId) == 13)

    genroundtrips(events.Electron.matched_gen)

    crossref(events[ak.num(events.Jet) > 2])
    crossref(events)

    assert ak.any(events.Photon.isTight, axis=1).tolist()[:9] == [False, True, True, True, False, False, False, False, False]


def test_read_nanodata():
    factory = NanoEventsFactory.from_file(os.path.abspath('tests/samples/nano_dimuon.root'))
    events = factory.events()

    crossref(events)
    crossref(events[ak.num(events.Jet) > 2])
