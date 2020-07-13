from coffea.nanoevents.methods.mixin import mixin_class
from coffea.nanoevents.methods.base import NanoCollection
from coffea.nanoevents.methods.vector import PtEtaPhiMLorentzVector


@mixin_class
class PtEtaPhiMCollection(PtEtaPhiMLorentzVector, NanoCollection):
    pass
