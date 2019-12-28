import numpy
import awkward
import numba
from .common import LorentzVector, Candidate
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
            if parent >= len(parent_all):
                raise RuntimeError("parent index beyond length of array!")
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
def _find_children(offsets_in, parentidx):
    offsets1_out = numpy.empty(len(parentidx) + 1, dtype=parentidx.dtype)
    content1_out = numpy.empty(len(parentidx), dtype=parentidx.dtype)
    offsets1_out[0] = 0

    offset0 = 0
    offset1 = 0
    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(stop_src - start_src):
            for possible_child in range(index, stop_src - start_src):
                if parentidx[start_src + possible_child] == index:
                    content1_out[offset1] = start_src + possible_child
                    offset1 = offset1 + 1
                    if offset1 >= len(content1_out):
                        raise RuntimeError("offset1 went out of bounds!")
            offsets1_out[offset0 + 1] = offset1
            offset0 = offset0 + 1
            if offset0 >= len(offsets1_out):
                raise RuntimeError("offset0 went out of bounds!")

    return offsets1_out, content1_out[:offset1]


class GenParticle(LorentzVector):
    '''NanoAOD generator-level particle object, including parent and child self-references

    Parent and child self-references are constructed from the ``genPartIdxMother`` column, where
    for each entry, the mother entry index is recorded, or -1 if no mother exists.
    '''
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
    '''bit-packed statusFlags interpretations.  Use `GenParticle.hasFlags` to query'''
    enable_children = True
    'Enable construction with children column (can be turned off if causing trouble)'

    def _finalize(self, name, events):
        parent_type = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(self.type.to.to))
        parent_type.check = False  # break recursion
        gen_parent = type(self)(
            self._lazy_crossref,
            args=(self._getcolumn('genPartIdxMother'), self),
            type=parent_type,
        )
        gen_parent.__doc__ = self.__doc__
        self['parent'] = gen_parent

        if self.enable_children:
            child_type = awkward.type.ArrayType(float('inf'), float('inf'), self.type.to.to)
            child_type.check = False
            children = type(self)(
                self._lazy_findchildren,
                args=(self._getcolumn('genPartIdxMother'),),
                type=child_type,
            )
            children.__doc__ = self.__doc__
            self['children'] = children

        self.type.check = False
        del self['genPartIdxMother']

    def hasFlags(self, flags):
        '''Check if one or more status flags are set

        Parameters
        ----------
            flags : list of str
                A list of flags that are required to be set true. Possible flags are enumerated in the `FLAGS` attribute

        Returns a boolean array
        '''
        dtype = self.statusFlags.type
        while not isinstance(dtype, numpy.dtype):
            try:
                dtype = dtype.to
            except AttributeError:
                dtype = dtype.type
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
        if isinstance(array, awkward.VirtualArray):
            array = array.array
        if isinstance(array, awkward.IndexedMaskedArray):
            mask = array.mask
            pdg_self = array.pdgId.fillna(0)
        else:
            mask = None
            pdg_self = numpy.array(array.pdgId)
        parent = array.parent
        if isinstance(array.parent, awkward.VirtualArray):
            parent = parent.array
        if not isinstance(parent, awkward.IndexedMaskedArray):
            raise RuntimeError
        parent_self = parent.mask
        if mask is not None:
            parent_self = numpy.where(mask >= 0, parent_self, -1)
        if isinstance(parent.content, awkward.VirtualArray):
            parent = parent.content.array
        pdg_all = numpy.array(parent.content.pdgId)
        parent_all = parent.content['_xref_%s_index' % self.rowname]
        globalindex = _find_distinctParent(pdg_self, pdg_all, parent_self, parent_all)
        out = type(parent)(
            globalindex,
            parent.content,
        )
        if jagged is not None:
            out = jagged.copy(content=out)
        return out

    def _lazy_findchildren(self, motherindices):
        # repair awkward type now that we've materialized
        motherindices.type.takes = self.array.offsets[-1]
        JaggedArray = self._get_mixin(self._get_methods(), awkward.JaggedArray)
        motherindices = motherindices.array
        offsets1, content1 = _find_children(
            self.array.offsets,
            motherindices.flatten()
        )
        return JaggedArray.fromoffsets(
            offsets1,
            content=self.array.content[content1]
        )


class GenVisTau(Candidate):
    '''NanoAOD visible tau object'''
    def _finalize(self, name, events):
        parent_type = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(events.GenPart.type.to.to))
        parent_type.check = False  # break recursion
        gen_parent = type(events.GenPart)(
            self._lazy_crossref,
            args=(self._getcolumn('genPartIdxMother'), events.GenPart),
            type=parent_type,
        )
        gen_parent.__doc__ = self.__doc__
        self['parent'] = gen_parent
        self.type.check = False
        del self['genPartIdxMother']
