import awkward1
from coffea.nanoevents.methods.util import apply_global_index
from coffea.nanoevents.methods.mixin import mixin_class, mixin_method
from coffea.nanoevents.methods.base import NanoCollection
from coffea.nanoevents.methods.vector import PtEtaPhiMLorentzVector
from coffea.nanoevents.methods.candidate import PtEtaPhiMCandidate


@mixin_class
class GenParticle(PtEtaPhiMLorentzVector, NanoCollection):
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


@mixin_class
class GenVisTau(PtEtaPhiMCandidate, NanoCollection):
    """NanoAOD visible tau object"""

    @property
    def parent(self):
        """Accessor to the parent particle"""
        return apply_global_index(self.genPartIdxMotherG, self._events().GenPart)
