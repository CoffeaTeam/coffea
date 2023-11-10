def max_chunks(fileset, maxchunks=None):
    return slice_chunks(fileset, slice(maxchunks))


def slice_chunks(fileset, theslice=slice(None)):
    if not isinstance(theslice, slice):
        theslice = slice(theslice)

    out = fileset.copy()
    for name, entry in fileset.items():
        for fname, finfo in entry["files"].items():
            out[name]["files"][fname]["steps"] = finfo["steps"][theslice]

    return out
