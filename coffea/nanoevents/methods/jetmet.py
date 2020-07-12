import awkward1
from coffea.nanoevents.methods.util import apply_global_index
from coffea.nanoevents.methods.mixin import mixin_class, mixin_method
from coffea.nanoevents.methods.base import NanoCollection
from coffea.nanoevents.methods.vector import PtEtaPhiMLorentzVector, PolarTwoVector


@mixin_class
class Jet(PtEtaPhiMLorentzVector, NanoCollection):
    """NanoAOD narrow radius jet object"""

    _enable_genjet = False
    "Set to true if using NanoAODv6 or newer (v5 had a bug in the mapping)"
    LOOSE = 0
    "jetId bit position"
    TIGHT = 1
    "jetId bit position"
    TIGHTLEPVETO = 2
    "jetId bit position"

    @property
    def isLoose(self):
        """Returns a boolean array marking loose jets according to jetId index"""
        return (self.jetId & (1 << self.LOOSE)) != 0

    @property
    def isTight(self):
        """Returns a boolean array marking tight jets according to jetId index"""
        return (self.jetId & (1 << self.TIGHT)) != 0

    @property
    def isTightLeptonVeto(self):
        """Returns a boolean array marking tight jets with explicit lepton veto according to jetId index"""
        return (self.jetId & (1 << self.TIGHTLEPVETO)) != 0

    @property
    def matched_electrons(self):
        return apply_global_index(self.electronIdxG, self._events().Electron)

    @property
    def matched_muons(self):
        return apply_global_index(self.muonIdxG, self._events().Muon)

    @property
    def matched_gen(self):
        return apply_global_index(self.genJetIdxG, self._events().GenJet)


@mixin_class
class FatJet(PtEtaPhiMLorentzVector, NanoCollection):
    """NanoAOD large radius jet object"""

    subjetmap = {
        "FatJet": "SubJet"
    }  # V6 has 'GenJetAK8': 'SubGenJetAK8', maybe better to put in generator.py
    LOOSE = 0
    "jetId bit position"
    TIGHT = 1
    "jetId bit position"
    TIGHTLEPVETO = 2
    "jetId bit position"

    @property
    def isLoose(self):
        """Returns a boolean array marking loose jets according to jetId index"""
        return (self.jetId & (1 << self.LOOSE)) != 0

    @property
    def isTight(self):
        """Returns a boolean array marking tight jets according to jetId index"""
        return (self.jetId & (1 << self.TIGHT)) != 0

    @property
    def isTightLeptonVeto(self):
        """Returns a boolean array marking tight jets with explicit lepton veto according to jetId index"""
        return (self.jetId & (1 << self.TIGHTLEPVETO)) != 0

    @property
    def subjets(self):
        return apply_global_index(self.subJetIdxG, self._events().SubJet)


@mixin_class
class MissingET(PolarTwoVector, NanoCollection):
    @property
    def r(self):
        return self["pt"]
