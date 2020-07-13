import awkward1
from coffea.nanoevents.methods.util import apply_global_index
from coffea.nanoevents.methods.mixin import mixin_class, mixin_method
from coffea.nanoevents.methods.base import NanoCollection
from coffea.nanoevents.methods.candidate import PtEtaPhiMCandidate


class CommonMatched:
    @property
    def matched_gen(self):
        return apply_global_index(self.genPartIdxG, self._events().GenPart)

    @property
    def matched_jet(self):
        return apply_global_index(self.jetIdxG, self._events().Jet)


@mixin_class
class Electron(PtEtaPhiMCandidate, NanoCollection, CommonMatched):
    """NanoAOD electron object"""

    FAIL = 0
    "cutBased selection minimum value"
    VETO = 1
    "cutBased selection minimum value"
    LOOSE = 2
    "cutBased selection minimum value"
    MEDIUM = 3
    "cutBased selection minimum value"
    TIGHT = 4
    "cutBased selection minimum value"
    pass

    @property
    def isVeto(self):
        """Returns a boolean array marking veto cut-based photons"""
        return (self.cutBased & (1 << self.VETO)) != 0

    @property
    def isLoose(self):
        """Returns a boolean array marking loose cut-based photons"""
        return (self.cutBased & (1 << self.LOOSE)) != 0

    @property
    def isMedium(self):
        """Returns a boolean array marking medium cut-based photons"""
        return (self.cutBased & (1 << self.MEDIUM)) != 0

    @property
    def isTight(self):
        """Returns a boolean array marking tight cut-based photons"""
        return (self.cutBased & (1 << self.TIGHT)) != 0

    @property
    def matched_photon(self):
        return apply_global_index(self.photonIdxG, self._events().Photon)


@mixin_class
class Muon(PtEtaPhiMCandidate, NanoCollection, CommonMatched):
    """NanoAOD muon object"""


@mixin_class
class Tau(PtEtaPhiMCandidate, NanoCollection, CommonMatched):
    """NanoAOD tau object"""


@mixin_class
class Photon(PtEtaPhiMCandidate, NanoCollection, CommonMatched):
    """NanoAOD photon object"""

    LOOSE = 0
    "cutBasedBitmap bit position"
    MEDIUM = 1
    "cutBasedBitmap bit position"
    TIGHT = 2
    "cutBasedBitmap bit position"

    @property
    def mass(self):
        return awkward1.broadcast_arrays(self.pt, 0.0)[1]

    @property
    def isLoose(self):
        """Returns a boolean array marking loose cut-based photons"""
        return (self.cutBasedBitmap & (1 << self.LOOSE)) != 0

    @property
    def isMedium(self):
        """Returns a boolean array marking medium cut-based photons"""
        return (self.cutBasedBitmap & (1 << self.MEDIUM)) != 0

    @property
    def isTight(self):
        """Returns a boolean array marking tight cut-based photons"""
        return (self.cutBasedBitmap & (1 << self.TIGHT)) != 0

    @property
    def matched_electron(self):
        return apply_global_index(self.electronIdxG, self._events().Electron)
