import awkward1
from coffea.nanoevents.methods import base, vector, candidate
from coffea.nanoevents.methods.util import apply_global_index


behavior = {}
behavior.update(base.behavior)
# vector behavior is included in candidate behavior
behavior.update(candidate.behavior)


@awkward1.mixin_class(behavior)
class PtEtaPhiMCollection(vector.PtEtaPhiMLorentzVector, base.NanoCollection):
    """Generic collection that has Lorentz vector properties"""

    pass


@awkward1.mixin_class(behavior)
class GenParticle(vector.PtEtaPhiMLorentzVector, base.NanoCollection):
    """NanoAOD generator-level particle object, including parent and child self-references

    Parent and child self-references are constructed from the ``genPartIdxMother`` column, where
    for each entry, the mother entry index is recorded, or -1 if no mother exists.
    """

    FLAGS = [
        "isPrompt",
        "isDecayedLeptonHadron",
        "isTauDecayProduct",
        "isPromptTauDecayProduct",
        "isDirectTauDecayProduct",
        "isDirectPromptTauDecayProduct",
        "isDirectHadronDecayProduct",
        "isHardProcess",
        "fromHardProcess",
        "isHardProcessTauDecayProduct",
        "isDirectHardProcessTauDecayProduct",
        "fromHardProcessBeforeFSR",
        "isFirstCopy",
        "isLastCopy",
        "isLastCopyBeforeFSR",
    ]
    """bit-packed statusFlags interpretations.  Use `GenParticle.hasFlags` to query"""

    def hasFlags(self, *flags):
        """Check if one or more status flags are set

        Parameters
        ----------
            flags : str or list
                A list of flags that are required to be set true. If the first argument
                is a list, it is expanded and subsequent arguments ignored.
                Possible flags are enumerated in the `FLAGS` attribute

        Returns a boolean array
        """
        if not len(flags):
            raise ValueError("No flags specified")
        elif isinstance(flags[0], list):
            flags = flags[0]
        mask = 0
        for flag in flags:
            mask |= 1 << self.FLAGS.index(flag)
        return (self.statusFlags & mask) == mask

    @property
    def parent(self):
        """Accessor to the parent particle"""
        return apply_global_index(self.genPartIdxMotherG, self._events().GenPart)

    @property
    def distinctParent(self):
        return apply_global_index(self.distinctParentIdxG, self._events().GenPart)

    @property
    def children(self):
        return apply_global_index(self.childrenIdxG, self._events().GenPart)

    @property
    def distinctChildren(self):
        return apply_global_index(self.distinctChildrenIdxG, self._events().GenPart)


@awkward1.mixin_class(behavior)
class GenVisTau(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD visible tau object"""

    @property
    def parent(self):
        """Accessor to the parent particle"""
        return apply_global_index(self.genPartIdxMotherG, self._events().GenPart)


@awkward1.mixin_class(behavior)
class Electron(candidate.PtEtaPhiMCandidate, base.NanoCollection):
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
    def matched_gen(self):
        return apply_global_index(self.genPartIdxG, self._events().GenPart)

    @property
    def matched_jet(self):
        return apply_global_index(self.jetIdxG, self._events().Jet)

    @property
    def matched_photon(self):
        return apply_global_index(self.photonIdxG, self._events().Photon)


@awkward1.mixin_class(behavior)
class Muon(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD muon object"""

    @property
    def matched_fsrPhoton(self):
        return apply_global_index(self.fsrPhotonIdxG, self._events().FsrPhoton)

    @property
    def matched_gen(self):
        return apply_global_index(self.genPartIdxG, self._events().GenPart)

    @property
    def matched_jet(self):
        return apply_global_index(self.jetIdxG, self._events().Jet)


@awkward1.mixin_class(behavior)
class Tau(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD tau object"""

    @property
    def matched_gen(self):
        return apply_global_index(self.genPartIdxG, self._events().GenPart)

    @property
    def matched_jet(self):
        return apply_global_index(self.jetIdxG, self._events().Jet)


@awkward1.mixin_class(behavior)
class Photon(candidate.PtEtaPhiMCandidate, base.NanoCollection):
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

    @property
    def matched_gen(self):
        return apply_global_index(self.genPartIdxG, self._events().GenPart)

    @property
    def matched_jet(self):
        return apply_global_index(self.jetIdxG, self._events().Jet)


@awkward1.mixin_class(behavior)
class FsrPhoton(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD fsr photon object"""

    @property
    def matched_muon(self):
        return apply_global_index(self.muonIdxG, self._events().Muon)


@awkward1.mixin_class(behavior)
class Jet(vector.PtEtaPhiMLorentzVector, base.NanoCollection):
    """NanoAOD narrow radius jet object"""

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


@awkward1.mixin_class(behavior)
class FatJet(vector.PtEtaPhiMLorentzVector, base.NanoCollection):
    """NanoAOD large radius jet object"""

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


@awkward1.mixin_class(behavior)
class MissingET(vector.PolarTwoVector, base.NanoCollection):
    """NanoAOD Missing transverse energy object"""

    @property
    def r(self):
        return self["pt"]
