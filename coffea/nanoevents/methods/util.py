import numpy
import awkward1


def apply_local_index(index, target):
    index = index + target._starts()

    def flat_take(layout, depth):
        if layout.purelist_depth == 1:
            return lambda: target.layout.content[layout]

    out = awkward1._util.recursively_apply(index.layout, flat_take)
    return awkward1._util.wrap(out, target.behavior)


def get_crossref(index, target):
    index = index.mask[index >= 0]
    return apply_local_index(index, target)
