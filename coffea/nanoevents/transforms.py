import copy
import numpy
import numba
import awkward1
from coffea.nanoevents.util import concat


def offsets(stack):
    """Extract offsets from ListOffsetArray

    Signature: array,!offsets
    """
    stack.append(stack.pop().layout.offsets)


def mask(stack):
    """Extract mask from a masked array

    Signature: array,!mask
    """
    stack.append(stack.pop().layout.mask)


def index(stack):
    """Extract index from array

    Signature: array,!index
    """
    stack.append(stack.pop().layout.index)


def starts(stack):
    """Extract offsets from ListArray

    Signature: array,!offsets
    """
    stack.append(stack.pop().layout.starts)


def stops(stack):
    """Extract offsets from ListArray

    Signature: array,!stops
    """
    stack.append(stack.pop().layout.stops)


def tags(stack):
    """Extract tags from UnionArray

    Signature: array,!tags
    """
    stack.append(stack.pop().layout.tags)


def content(stack):
    """Extract content from array

    Signature: array,!content
    """
    stack.append(stack.pop().layout.content)


def counts2offsets_form(counts_form):
    form = {
        "class": "NumpyArray",
        "itemsize": 8,
        "format": "i",
        "primitive": "int64",
        "parameters": counts_form.get("parameters", None),
        "form_key": concat(counts_form["form_key"], "!counts2offsets"),
    }
    return form


def counts2offsets(stack):
    """Cumulative sum of counts

    Signature: counts,!counts2offsets
    Outputs an array with length one larger than input
    """
    counts = numpy.array(stack.pop())
    offsets = numpy.empty(len(counts) + 1, dtype=numpy.int64)
    offsets[0] = 0
    numpy.cumsum(counts, out=offsets[1:])
    stack.append(offsets)


def local2global_form(index, target_offsets):
    if not index["class"].startswith("ListOffset"):
        raise RuntimeError
    if not target_offsets["class"] == "NumpyArray":
        raise RuntimeError
    form = copy.deepcopy(index)
    form["content"]["form_key"] = concat(
        index["form_key"], target_offsets["form_key"], "!local2global"
    )
    form["content"]["itemsize"] = 8
    form["content"]["primitive"] = "int64"
    return form


def local2global(stack):
    """Turn jagged local index into global index

    Signature: index,target_offsets,!local2global
    Outputs a content array with same shape as index content
    """
    target_offsets = stack.pop()
    index = stack.pop()
    index = index.mask[index >= 0] + target_offsets[:-1]
    out = numpy.array(awkward1.flatten(awkward1.fill_none(index, -1)))
    if out.dtype != numpy.int64:
        raise RuntimeError
    stack.append(out)


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


def distinctParent_form(parents, pdg):
    if not parents["class"].startswith("ListOffset"):
        raise RuntimeError
    if not pdg["class"].startswith("ListOffset"):
        raise RuntimeError
    form = {
        "class": "ListOffsetArray64",
        "offsets": "i64",
        "content": {
            "class": "NumpyArray",
            "itemsize": 8,
            "format": "i",
            "primitive": "int64",
        },
        "form_key": parents["form_key"],
    }
    form["content"]["form_key"] = concat(
        parents["content"]["form_key"], pdg["content"]["form_key"], "!distinctParent",
    )
    return form


def distinctParent(stack):
    """Compute first parent with distinct PDG id

    Signature: globalparents,globalpdgs,!distinctParent
    Expects global indexes, flat arrays, which should be same length
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


def children_form(offsets, globalparents):
    if not globalparents["class"].startswith("ListOffset"):
        raise RuntimeError
    form = {
        "class": "ListOffsetArray64",
        "offsets": "i64",
        "content": {
            "class": "ListOffsetArray64",
            "offsets": "i64",
            "content": {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "i",
                "primitive": "int64",
            },
        },
    }
    form["form_key"] = offsets["form_key"]
    key = concat(offsets["form_key"], globalparents["content"]["form_key"], "!children")
    form["content"]["form_key"] = key
    form["content"]["content"]["form_key"] = concat(key, "!content")
    return form


def children(stack):
    """Compute children

    Signature: offsets,globalparents,!children
    Output will be a jagged array with same outer shape as globalparents content
    """
    parents = stack.pop()
    offsets = stack.pop()
    coffsets, ccontent = _children_kernel(offsets, parents)
    out = awkward1.Array(
        awkward1.layout.ListOffsetArray64(
            awkward1.layout.Index64(coffsets), awkward1.layout.NumpyArray(ccontent),
        )
    )
    stack.append(out)


def nestedindex_form(indices):
    if not all(index["class"].startswith("ListOffset") for index in indices):
        raise RuntimeError
    form = {
        "class": "ListOffsetArray64",
        "offsets": "i64",
        "content": copy.deepcopy(indices[0]),
    }
    # steal offsets from first input
    key = []
    for index in indices:
        key.append(index["content"]["form_key"])
    key.append("!nestedindex")
    key = concat(*key)
    form["form_key"] = indices[0]["form_key"]
    form["content"]["form_key"] = key
    form["content"]["content"]["form_key"] = concat(key, "!content")
    return form


def nestedindex(stack):
    """Concatenate a list of indices along a new axis

    Signature: index1,index2,...,!nestedindex
    Index arrays should all be same shape flat arrays
    Outputs a jagged array with same outer shape as index arrays
    """
    indexers = stack[:]
    stack.clear()
    # return awkward1.concatenate([idx[:, None] for idx in indexers], axis=1)
    n = len(indexers)
    out = numpy.empty(n * len(indexers[0]), dtype="int64")
    for i, idx in enumerate(indexers):
        out[i::n] = idx
    offsets = numpy.arange(0, len(out) + 1, n, dtype=numpy.int64)
    out = awkward1.Array(
        awkward1.layout.ListOffsetArray64(
            awkward1.layout.Index64(offsets), awkward1.layout.NumpyArray(out),
        )
    )
    stack.append(out)
