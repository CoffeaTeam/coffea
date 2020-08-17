from cachetools import LRUCache
from collections.abc import Mapping
import numpy
import uproot4
import coffea.nanoevents.transforms as transforms
from coffea.nanoevents.util import key_to_tuple, tuple_to_key


class UprootSourceMapping(Mapping):
    def __init__(self, uuid_pfnmap, uproot_options, cachesize=1):
        self._uuid_pfnmap = uuid_pfnmap
        self._uproot_options = uproot_options
        self._cachesize = cachesize
        self._cache = LRUCache(self._cachesize)

    def __getstate__(self):
        return {
            "uuid_pfnmap": self._uuid_pfnmap,
            "uproot_options": self._uproot_options,
        }

    def __setstate__(self, state):
        self._uuid_pfnmap = state["uuid_pfnmap"]
        self._uproot_options = state["uproot_options"]
        self._cachesize = state["cachesize"]
        self._cache = LRUCache(self._cachesize)

    def _tree(self, uuid, treepath):
        try:
            return self._cache[uuid][treepath]
        except KeyError:
            pass
        rootdir = uproot4.open(self._uuid_pfnmap[uuid], **self._uproot_options)
        if str(rootdir.file.uuid) != uuid:
            raise RuntimeError("PFN UUID does not match expected value")
        self._cache[uuid] = rootdir
        return rootdir[treepath]

    def __getitem__(self, key):
        uuid, treepath, entryrange, form_key, *layoutattr = key_to_tuple(key)
        start, stop = (int(x) for x in entryrange.split("-"))
        print("gettting:", uuid, treepath, start, stop, form_key, layoutattr)
        nodes = form_key.split(",")
        if len(layoutattr) == 1:
            nodes.append("!" + layoutattr[0])
        elif len(layoutattr) > 1:
            raise RuntimeError
        stack = []
        skip = False
        for node in nodes:
            if skip:
                skip = False
                continue
            elif node == "!skip":
                skip = True
                continue
            elif node == "!load":
                branch = self._tree(uuid, treepath)[stack.pop()]
                stack.append(branch.array(entry_start=start, entry_stop=stop))
            elif node.startswith("!"):
                tname = node[1:]
                if not hasattr(transforms, tname):
                    raise RuntimeError(
                        f"Syntax error in form_key: no transform named {tname}"
                    )
                getattr(transforms, tname)(stack)
            else:
                stack.append(node)
        if len(stack) != 1:
            raise RuntimeError(f"Syntax error in form_key {form_key}")
        out = numpy.array(stack.pop())
        if out.dtype == numpy.object:
            raise RuntimeError(f"Left with non-bare array after evaluating {form_key}")
        return out

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
