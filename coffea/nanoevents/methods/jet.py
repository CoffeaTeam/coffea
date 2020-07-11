import awkward1
from coffea.nanoevents.methods.util import get_crossref
from coffea.nanoevents.methods.mixin import mixin_class, mixin_method
from coffea.nanoevents.methods.base import NanoCollection
from coffea.nanoevents.methods.vector import PtEtaPhiMLorentzVector


@mixin_class
class Jet(PtEtaPhiMLorentzVector, NanoCollection):
    pass


@mixin_class
class FatJet(PtEtaPhiMLorentzVector, NanoCollection):
    pass
