import numpy
import awkward
import numba
from .common import LorentzVector
from ..util import _mixin


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


@numba.njit
def _resize(a, new_size):
    new = numpy.empty(new_size, a.dtype)
    new[:a.size] = a
    return new


@numba.njit
def _find_children(offsets_src, localindex_dst):
    offsets1_out = numpy.empty(len(localindex_dst) + 1, dtype=localindex_dst.dtype)
    content1_out = numpy.empty(len(localindex_dst), dtype=localindex_dst.dtype)
    offsets1_out[0] = 0

    offset0 = 0
    offset1 = 0
    for record_index in range(len(offsets_src) - 1):
        start_src, stop_src = offsets_src[record_index], offsets_src[record_index + 1]

        for index1 in range(stop_src - start_src):
            for index_out in range(index1, stop_src - start_src):
                if localindex_dst[index_out] == index1:
                    content1_out[offset1] = index_out
                    offset1 = offset1 + 1
            offsets1_out[offset0 + 1] = offset1
            offset0 = offset0 + 1

    return offsets1_out[:offset0 + 1], content1_out[:offset1]


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
        '''Check if one or more flags are set

        Possible flags are enumerated in the FLAGS attribute
        '''
        dtype = self.statusFlags.type
        while not isinstance(dtype, numpy.dtype):
            dtype = dtype.to
        bits = numpy.array([self.FLAGS.index(f) for f in flags])
        mask = (1 << bits).sum().astype(dtype)
        return (self.statusFlags & mask) == mask

    @property
    def distinctParent(self):
        '''Find the particle's parent, skipping ancestors with the same pdgId'''
        array = self
        jagged = None
        if isinstance(array, awkward.VirtualArray):
            array = array.array
        if isinstance(array, awkward.JaggedArray):
            jagged = array
            array = array.content
        pdg_self = numpy.array(array.pdgId)
        parent = array.parent
        if isinstance(array.parent, awkward.VirtualArray):
            parent = parent.array
        if not isinstance(parent, awkward.IndexedMaskedArray):
            raise RuntimeError
        parent_self = parent.mask
        while isinstance(parent.content, awkward.IndexedMaskedArray):
            parent = parent.content
        pdg_all = numpy.array(parent.content.pdgId)
        parent_all = parent.mask
        globalindex = _find_distinctParent(pdg_self, pdg_all, parent_self, parent_all)
        out = type(parent)(
            globalindex,
            parent.content,
        )
        if jagged is not None:
            out = jagged.copy(content=out)
        return out

    def _lazy_findchildren(self, motherindices):
        motherindices = motherindices.array
        offsets1, content1 = _find_children(
            self.array.offsets,
            motherindices.flatten()
        )
        return awkward.JaggedArray.fromoffsets(
            offsets1,
            content=self.array.content[content1]
        )
