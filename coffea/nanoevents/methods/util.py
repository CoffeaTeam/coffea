import awkward1


def apply_global_index(index, target):
    def flat_take(layout):
        idx = awkward1.Array(layout)
        return target._content()[idx.mask[idx >= 0]]

    def descend(layout, depth):
        if layout.purelist_depth == 1:
            return lambda: flat_take(layout)

    (index,) = awkward1.broadcast_arrays(index)
    out = awkward1._util.recursively_apply(index.layout, descend)
    return awkward1._util.wrap(out, target.behavior)
