import numpy
import numba
import awkward1
from coffea.nanoevents.methods.util import get_crossref
from coffea.nanoevents.methods.mixin import mixin_class, mixin_method
from coffea.nanoevents.methods.base import NanoCollection
from coffea.nanoevents.methods.vector import PtEtaPhiMLorentzVector


@numba.njit
def _distinctParent_kernel(part_pdg, part_parent, allpart_pdg, allpart_parent):
    out = numpy.empty(len(part_pdg), dtype=numpy.int64)
    for i in range(len(part_pdg)):
        parent = part_parent[i]
        if parent < 0:
            out[i] = -1
            continue
        thispdg = part_pdg[i]
        while parent >= 0 and allpart_pdg[parent] == thispdg:
            if parent >= len(allpart_pdg):
                raise RuntimeError("parent index beyond length of array!")
            parent = allpart_parent[parent]
        out[i] = parent
    return out


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
        return get_crossref(self.genPartIdxMother, self._events().GenPart)

    @property
    def distinctParent(self):
        genp = self._events().GenPart
        allpart_pdg = awkward1.flatten(genp.pdgId)
        (allpart_pdg,) = awkward1.broadcast_arrays(allpart_pdg)  # force materialization
        allpart_parent = awkward1.flatten(
            (genp.genPartIdxMother >= 0) * (genp.genPartIdxMother + genp._starts() + 1) - 1
        )

        def take(particles):
            idx = _distinctParent_kernel(
                particles["pdg"], particles["parent"], allpart_pdg, allpart_parent
            )
            return awkward1.layout.IndexedOptionArray64(
                awkward1.layout.Index64(idx), genp.layout.content
            )

        def fcn(layout, depth):
            if layout.purelist_depth == 1:
                particles = awkward1._util.wrap(layout, None)
                return lambda: take(particles)

        parent_self = (self.genPartIdxMother >= 0) * (
            self.genPartIdxMother + genp._starts() + 1
        ) - 1
        trimmed = awkward1.zip({"pdg": self.pdgId, "parent": parent_self})
        (trimmed,) = awkward1.broadcast_arrays(trimmed)
        out = awkward1._util.recursively_apply(trimmed.layout, fcn)
        return awkward1._util.wrap(out, None)
