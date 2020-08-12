import numpy
import numba
import awkward1


def offsets(stack):
    """Extract offsets from ListOffsetArray

    Signature: array,!offsets
    """
    stack.append(stack.pop().layout.offsets)


def content(stack):
    """Extract content from ListOffsetArray

    Signature: array,!content
    """
    stack.append(stack.pop().layout.content)


def counts2offsets(stack):
    """Cumulative sum of counts

    Signature: counts,!counts2offsets
    Outputs an array with length one larger than input
    """
    counts = stack.pop()
    offsets = numpy.empty(len(counts) + 1, counts.dtype)
    offsets[0] = 0
    stack.push(numpy.cumsum(counts, out=offsets[1:]))


def local2global(stack):
    """Turn jagged local index into global index

    Signature: index,target_offsets,!local2global
    Outputs a jagged array with same shape as index
    """
    target_offsets = stack.pop()
    index = stack.pop()
    index = index.mask[index >= 0] + target_offsets[:-1]
    stack.append(awkward1.fill_none(index, -1))


@numba.njit
def _distinctParent_kernel(allpart_parent, allpart_pdg):
    out = numpy.empty(len(allpart_pdg), dtype=numpy.int64)
    for i in range(len(allpart_pdg)):
        parent = allpart_parent[i]
        if parent < 0:
            out[i] = -1
            continue
        thispdg = allpart_pdg[i]
        while parent >= 0 and allpart_pdg[parent] == thispdg:
            if parent >= len(allpart_pdg):
                raise RuntimeError("parent index beyond length of array!")
            parent = allpart_parent[parent]
        out[i] = parent
    return out


def distinctParent(stack):
    """Compute first parent with distinct PDG id

    Signature: globalparents,globalpdgs,!distinctParent
    Expects global indexes, arrays should be same length
    """
    pdg = stack.pop()
    parents = stack.pop()
    stack.append(_distinctParent_kernel(awkward1.Array(parents), awkward1.Array(pdg)))


@numba.njit
def _children_kernel(offsets_in, parentidx):
    offsets1_out = numpy.empty(len(parentidx) + 1, dtype=numpy.int64)
    content1_out = numpy.empty(len(parentidx), dtype=numpy.int64)
    offsets1_out[0] = 0

    offset0 = 0
    offset1 = 0
    for record_index in range(len(offsets_in) - 1):
        start_src, stop_src = offsets_in[record_index], offsets_in[record_index + 1]

        for index in range(start_src, stop_src):
            for possible_child in range(index, stop_src):
                if parentidx[possible_child] == index:
                    content1_out[offset1] = possible_child
                    offset1 = offset1 + 1
                    if offset1 >= len(content1_out):
                        raise RuntimeError("offset1 went out of bounds!")
            offsets1_out[offset0 + 1] = offset1
            offset0 = offset0 + 1
            if offset0 >= len(offsets1_out):
                raise RuntimeError("offset0 went out of bounds!")

    return offsets1_out, content1_out[:offset1]


def children(stack):
    """Compute children

    Signature: offsets,globalparents,!children
    Output will be a jagged array with same outer shape as offsets
    """
    parents = stack.pop()
    offsets = stack.pop()
    coffsets, ccontent = _children_kernel(
        awkward1.Array(offsets.array), awkward1.Array(parents.array)
    )
    stack.append(
        awkward1.layout.ListOffsetArray64(
            awkward1.layout.Index64(coffsets), awkward1.layout.NumpyArray(ccontent),
        )
    )


def nestedindex(stack):
    """Concatenate a list of indices

    Signature: index1,index2,...,!nestedindex
    Index arrays should all be same length
    Returns a jagged array
    """
    indexers, stack = stack[:], []
    # idx = awkward1.concatenate([idx[:, None] for idx in indexers], axis=1)
    n = len(indexers)
    out = numpy.empty(n * len(indexers[0]), dtype="int64")
    for i, idx in enumerate(indexers):
        out[i::n] = idx
    offsets = numpy.arange(0, len(out) + 1, n, dtype=numpy.int64)
    stack.append(
        awkward1.layout.ListOffsetArray64(
            awkward1.layout.Index64(offsets), awkward1.layout.NumpyArray(out),
        )
    )
