import numpy
import numba
import awkward1


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


def distinctParent(parents, pdg):
    def gen():
        return _distinctParent_kernel(
            awkward1.Array(parents.array), awkward1.Array(pdg.array)
        )

    form = awkward1.forms.Form.fromjson('"int64"')
    return awkward1.layout.ArrayGenerator(
        gen, (), {}, form=form, length=parents.generator.length,
    )


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


def children(offsets, parents):
    def gen():
        coffsets, ccontent = _children_kernel(
            awkward1.Array(offsets.array), awkward1.Array(parents.array)
        )
        return awkward1.layout.ListOffsetArray64(
            awkward1.layout.Index64(coffsets), awkward1.layout.NumpyArray(ccontent),
        )

    form = awkward1.forms.Form.fromjson(
        '{"class": "ListOffsetArray64", "offsets": "i64", "content": "int64"}'
    )
    return awkward1.layout.ArrayGenerator(
        gen, (), {}, form=form, length=parents.generator.length,
    )
