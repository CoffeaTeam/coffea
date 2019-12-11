import numpy
import awkward
import numba
from .common import LorentzVector


@numba.njit
def _find_distinctParent(pdg_self, pdg_all, parent_self, parent_all):
    out = parent_self.copy()
    for i in range(len(out)):
        if out[i] < 0:
            continue
        thispdg = pdg_self[i]
        parent = parent_self[i]
        parentpdg = pdg_all[parent]
        while parent >= 0 and parentpdg == thispdg:
            parent = parent_all[parent]
            parentpdg = pdg_all[parent]
        out[i] = parent

    return out


class GenParticle(LorentzVector):
    FLAGS = [
        'isPrompt',
        'isDecayedLeptonHadron',
        'isTauDecayProduct',
        'isPromptTauDecayProduct',
        'isDirectTauDecayProduct',
        'isDirectPromptTauDecayProduct',
        'isDirectHadronDecayProduct',
        'isHardProcess',
        'fromHardProcess',
        'isHardProcessTauDecayProduct',
        'isDirectHardProcessTauDecayProduct',
        'fromHardProcessBeforeFSR',
        'isFirstCopy',
        'isLastCopy',
        'isLastCopyBeforeFSR'
    ]

    def hasFlags(self, flags):
        dtype = self.statusFlags.type
        while not isinstance(dtype, numpy.dtype):
            dtype = dtype.to
        bits = numpy.array([self.FLAGS.index(f) for f in flags])
        mask = (1 << bits).sum().astype(dtype)
        return (self.statusFlags & mask) == mask

    @property
    def distinctParent(self):
        array = self
        jagged = None
        if isinstance(array, awkward.VirtualArray):
            array = array.array
        if isinstance(array, awkward.JaggedArray):
            jagged = array
            array = array.content
        pdg_self = numpy.array(array.pdgId)
        if isinstance(array.parent, awkward.VirtualArray):
            parent = array.parent.array
        else:
            parent = array.parent
        parent_self = parent.mask
        pdg_all = numpy.array(parent.content.pdgId)
        parent_all = parent.content['_%s_globalindex' % self.rowname]
        globalindex = _find_distinctParent(pdg_self, pdg_all, parent_self, parent_all)
        out = awkward.IndexedMaskedArray(
            globalindex,
            parent.content,
        )
        if jagged is not None:
            out = jagged.copy(content=out)
        return out
