"""Mixins for the CMS NanoAOD schema"""
import warnings

import awkward

from coffea.nanoevents.methods import base, candidate, vector

behavior = {}
behavior.update(base.behavior)
# vector behavior is included in candidate behavior
behavior.update(candidate.behavior)


class _NanoAODEvents(behavior["NanoEvents"]):
    def __repr__(self):
        return f"<event {getattr(self,'run','??')}:\
                {getattr(self,'luminosityBlock','??')}:\
                {getattr(self,'event','??')}>"


behavior["NanoEvents"] = _NanoAODEvents


def _set_repr_name(classname):
    def namefcn(self):
        return classname

    # behavior[("__typestr__", classname)] = classname[0].lower() + classname[1:]
    behavior[classname].__repr__ = namefcn


@awkward.mixin_class(behavior)
class PtEtaPhiMCollection(vector.PtEtaPhiMLorentzVector, base.NanoCollection):
    """Generic collection that has Lorentz vector properties"""

    pass


@awkward.mixin_class(behavior)
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
    def parent(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenPart
            return original._apply_global_index(
                _dask_array_.genPartIdxMotherG, _dask_array_=original
            )
        return self._events().GenPart._apply_global_index(self.genPartIdxMotherG)

    @property
    def distinctParent(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenPart
            return original._apply_global_index(
                _dask_array_.distinctParentIdxG, _dask_array_=original
            )
        return self._events().GenPart._apply_global_index(self.distinctParentIdxG)

    @property
    def children(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenPart
            return original._apply_global_index(
                _dask_array_.childrenIdxG, _dask_array_=original
            )
        return self._events().GenPart._apply_global_index(self.childrenIdxG)

    @property
    def distinctChildren(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenPart
            return original._apply_global_index(
                _dask_array_.distinctChildrenIdxG, _dask_array_=original
            )
        return self._events().GenPart._apply_global_index(self.distinctChildrenIdxG)

    @property
    def distinctChildrenDeep(self, _dask_array_=None):
        """Accessor to distinct child particles with different PDG id, or last ones in the chain"""
        warnings.warn(
            "distinctChildrenDeep may not give correct answers for all generators!"
        )
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenPart
            return original._apply_global_index(
                _dask_array_.distinctChildrenDeepIdxG, _dask_array_=original
            )
        return self._events().GenPart._apply_global_index(self.distinctChildrenDeepIdxG)


_set_repr_name("GenParticle")


@awkward.mixin_class(behavior)
class GenVisTau(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD visible tau object"""

    @property
    def parent(self, _dask_array_=None):
        """Accessor to the parent particle"""
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenPart
            return original._apply_global_index(
                _dask_array_.genPartIdxMotherG, _dask_array_=original
            )
        return self._events().GenPart._apply_global_index(self.genPartIdxMotherG)


_set_repr_name("GenVisTau")


@awkward.mixin_class(behavior)
class Electron(candidate.PtEtaPhiMCandidate, base.NanoCollection, base.Systematic):
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
        """Returns a boolean array marking veto cut-based electrons"""
        return self.cutBased >= self.VETO

    @property
    def isLoose(self):
        """Returns a boolean array marking loose cut-based electrons"""
        return self.cutBased >= self.LOOSE

    @property
    def isMedium(self):
        """Returns a boolean array marking medium cut-based electrons"""
        return self.cutBased >= self.MEDIUM

    @property
    def isTight(self):
        """Returns a boolean array marking tight cut-based electrons"""
        return self.cutBased >= self.TIGHT

    @property
    def matched_gen(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenPart
            return original._apply_global_index(
                _dask_array_.genPartIdxG, _dask_array_=original
            )
        return self._events().GenPart._apply_global_index(self.genPartIdxG)

    @property
    def matched_jet(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().Jet
            return original._apply_global_index(
                _dask_array_.jetIdxG, _dask_array_=original
            )
        return self._events().Jet._apply_global_index(self.jetIdxG)

    @property
    def matched_photon(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().Photon
            return original._apply_global_index(
                _dask_array_.photonIdxG, _dask_array_=original
            )
        return self._events().Photon._apply_global_index(self.photonIdxG)


_set_repr_name("Electron")


@awkward.mixin_class(behavior)
class Muon(candidate.PtEtaPhiMCandidate, base.NanoCollection, base.Systematic):
    """NanoAOD muon object"""

    @property
    def matched_fsrPhoton(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().FsrPhoton
            return original._apply_global_index(
                _dask_array_.fsrPhotonIdxG, _dask_array_=original
            )
        return self._events().FsrPhoton._apply_global_index(self.fsrPhotonIdxG)

    @property
    def matched_gen(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenPart
            return original._apply_global_index(
                _dask_array_.genPartIdxG, _dask_array_=original
            )
        return self._events().GenPart._apply_global_index(self.genPartIdxG)

    @property
    def matched_jet(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().Jet
            return original._apply_global_index(
                _dask_array_.jetIdxG, _dask_array_=original
            )
        return self._events().Jet._apply_global_index(self.jetIdxG)


_set_repr_name("Muon")


@awkward.mixin_class(behavior)
class Tau(candidate.PtEtaPhiMCandidate, base.NanoCollection, base.Systematic):
    """NanoAOD tau object"""

    @property
    def matched_gen(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenPart
            return original._apply_global_index(
                _dask_array_.genPartIdxG, _dask_array_=original
            )
        return self._events().GenPart._apply_global_index(self.genPartIdxG)

    @property
    def matched_jet(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().Jet
            return original._apply_global_index(
                _dask_array_.jetIdxG, _dask_array_=original
            )
        return self._events().Jet._apply_global_index(self.jetIdxG)


_set_repr_name("Tau")


@awkward.mixin_class(behavior)
class Photon(candidate.PtEtaPhiMCandidate, base.NanoCollection, base.Systematic):
    """NanoAOD photon object"""

    LOOSE = 0
    "cutBasedBitmap bit position"
    MEDIUM = 1
    "cutBasedBitmap bit position"
    TIGHT = 2
    "cutBasedBitmap bit position"

    @property
    def mass(self):
        return 0.0 * self.pt

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
    def matched_electron(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().Electron
            return original._apply_global_index(
                _dask_array_.electronIdxG, _dask_array_=original
            )
        return self._events().Electron._apply_global_index(self.electronIdxG)

    @property
    def matched_gen(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenPart
            return original._apply_global_index(
                _dask_array_.genPartIdxG, _dask_array_=original
            )
        return self._events().GenPart._apply_global_index(self.genPartIdxG)

    @property
    def matched_jet(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().Jet
            return original._apply_global_index(
                _dask_array_.jetIdxG, _dask_array_=original
            )
        return self._events().Jet._apply_global_index(self.jetIdxG)


_set_repr_name("Photon")


@awkward.mixin_class(behavior)
class FsrPhoton(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """NanoAOD fsr photon object"""

    @property
    def matched_muon(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().Jet
            return original._apply_global_index(
                _dask_array_.muonIdxG, _dask_array_=original
            )
        return self._events().Muon._apply_global_index(self.muonIdxG)


_set_repr_name("FsrPhoton")


@awkward.mixin_class(behavior)
class Jet(vector.PtEtaPhiMLorentzVector, base.NanoCollection, base.Systematic):
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
    def matched_electrons(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().Electron
            return original._apply_global_index(
                _dask_array_.electronIdxG, _dask_array_=original
            )
        return self._events().Electron._apply_global_index(self.electronIdxG)

    @property
    def matched_muons(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().Muon
            return original._apply_global_index(
                _dask_array_.muonIdxG, _dask_array_=original
            )
        return self._events().Muon._apply_global_index(self.muonIdxG)

    @property
    def matched_gen(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenJet
            return original._apply_global_index(
                _dask_array_.genJetIdxG, _dask_array_=original
            )
        return self._events().GenJet._apply_global_index(self.genJetIdxG)

    @property
    def constituents(self, _dask_array_=None):
        if "pFCandsIdxG" not in self.fields:
            raise RuntimeError("PF candidates are only available for PFNano")
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().JetPFCands
            return original._apply_global_index(
                _dask_array_.pFCandsIdxG, _dask_array_=original
            )
        return self._events().JetPFCands._apply_global_index(self.pFCandsIdxG)


_set_repr_name("Jet")


@awkward.mixin_class(behavior)
class FatJet(vector.PtEtaPhiMLorentzVector, base.NanoCollection, base.Systematic):
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
    def subjets(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().SubJet
            return original._apply_global_index(
                _dask_array_.subJetIdxG, _dask_array_=original
            )
        return self._events().SubJet._apply_global_index(self.subJetIdxG)

    @property
    def matched_gen(self, _dask_array_=None):
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().GenJetAK8
            return original._apply_global_index(
                _dask_array_.genJetAK8IdxG, _dask_array_=original
            )
        return self._events().GenJetAK8._apply_global_index(self.genJetAK8IdxG)

    @property
    def constituents(self, _dask_array_=None):
        if "pFCandsIdxG" not in self.fields:
            raise RuntimeError("PF candidates are only available for PFNano")
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]().FatJetPFCands
            return original._apply_global_index(
                _dask_array_.pFCandsIdxG, _dask_array_=original
            )
        return self._events().FatJetPFCands._apply_global_index(self.pFCandsIdxG)


_set_repr_name("FatJet")


@awkward.mixin_class(behavior)
class MissingET(vector.PolarTwoVector, base.NanoCollection, base.Systematic):
    """NanoAOD Missing transverse energy object"""

    @property
    def r(self):
        return self["pt"]


_set_repr_name("MissingET")


@awkward.mixin_class(behavior)
class Vertex(base.NanoCollection):
    """NanoAOD vertex object"""

    @property
    def pos(self):
        """Vertex position as a three vector"""
        return awkward.zip(
            {
                "x": self["x"],
                "y": self["y"],
                "z": self["z"],
            },
            with_name="ThreeVector",
            behavior=self.behavior,
        )


_set_repr_name("Vertex")


@awkward.mixin_class(behavior)
class SecondaryVertex(Vertex):
    """NanoAOD secondary vertex object"""

    @property
    def p4(self):
        """4-momentum vector of tracks associated to this SV"""
        return awkward.zip(
            {
                "pt": self["pt"],
                "eta": self["eta"],
                "phi": self["phi"],
                "mass": self["mass"],
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=self.behavior,
        )


_set_repr_name("SecondaryVertex")


@awkward.mixin_class(behavior)
class AssociatedPFCand(base.NanoCollection):
    """PFNano PF candidate to jet association object"""

    collection_map = {
        "JetPFCands": ("Jet", "PFCands"),
        "FatJetPFCands": ("FatJet", "PFCands"),
        "GenJetCands": ("GenJet", "GenCands"),
        "GenFatJetCands": ("GenJetAK8", "GenCands"),
    }

    @property
    def jet(self, _dask_array_=None):
        collection = self.collection_map[self._collection_name()][0]
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]()[collection]
            return original._apply_global_index(
                _dask_array_.jetIdxG, _dask_array_=original
            )
        return self._events()[collection]._apply_global_index(self.jetIdxG)

    @property
    def pf(self, _dask_array_=None):
        collection = self.collection_map[self._collection_name()][1]
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]()[collection]
            return original._apply_global_index(
                _dask_array_.pFCandsIdxG, _dask_array_=original
            )
        return self._events()[collection]._apply_global_index(self.pFCandsIdxG)


_set_repr_name("AssociatedPFCand")


@awkward.mixin_class(behavior)
class AssociatedSV(base.NanoCollection):
    """PFNano secondary vertex to jet association object"""

    collection_map = {
        "JetSVs": ("Jet", "SV"),
        "FatJetSVs": ("FatJet", "SV"),
        # these two are unclear
        "GenJetSVs": ("GenJet", "SV"),
        "GenFatJetSVs": ("GenJetAK8", "SV"),
    }

    @property
    def jet(self, _dask_array_=None):
        collection = self._events()[self.collection_map[self._collection_name()][0]]
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]()[collection]
            return original._apply_global_index(
                _dask_array_.jetIdxG, _dask_array_=original
            )
        return self._events()[collection]._apply_global_index(self.jetIdxG)

    @property
    def sv(self, _dask_array_=None):
        collection = self.collection_map[self._collection_name()][1]
        if _dask_array_ is not None:
            original = _dask_array_.behavior["__original_array__"]()[collection]
            return original._apply_global_index(
                _dask_array_.sVIdxG, _dask_array_=original
            )
        return self._events()[collection]._apply_global_index(self.sVIdxG)


_set_repr_name("AssociatedSV")


@awkward.mixin_class(behavior)
class PFCand(candidate.PtEtaPhiMCandidate, base.NanoCollection):
    """PFNano particle flow candidate object"""

    pass


_set_repr_name("PFCand")


__all__ = [
    "PtEtaPhiMCollection",
    "GenParticle",
    "GenVisTau",
    "Electron",
    "Muon",
    "Tau",
    "Photon",
    "FsrPhoton",
    "Jet",
    "FatJet",
    "MissingET",
    "Vertex",
    "SecondaryVertex",
    "AssociatedPFCand",
    "AssociatedSV",
    "PFCand",
]
