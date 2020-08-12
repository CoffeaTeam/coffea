from cachetools import LRUCache
from urllib.parse import quote, unquote
from collections.abc import Mapping
import numpy
import uproot4
import coffea.nanoevents.transforms as transforms


def tuple_to_key(tup):
    return "/".join(quote(x, safe="") for x in tup)


def key_to_tuple(key):
    return tuple(unquote(x) for x in key.split("/"))


class UprootSourceMapping(Mapping):
    def __init__(self, uuid_pfnmap, uproot_options, cachesize=1):
        self._uuid_pfnmap = uuid_pfnmap
        self._uproot_options = uproot_options
        self._cache = LRUCache(cachesize)

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
        uuid, treepath, entryrange, form_key, *nodeattrs = key_to_tuple(key)
        start, stop = (int(x) for x in entryrange.split("-"))
        print("gettting:", uuid, treepath, start, stop, form_key, nodeattrs)
        stack = []
        for node in form_key.split(","):
            if node == "!load":
                branch = self._tree(uuid, treepath)[stack.pop()]
                stack.append(branch.array(entry_start=start, entry_stop=stop))
            elif node.startswith("!"):
                tname = node[1:]
                if not hasattr(transforms, tname):
                    raise RuntimeError(
                        "Syntax error in form_key: no transform named", tname
                    )
                getattr(transforms, tname)(stack)
            else:
                stack.append(node)
        if len(stack) != 1:
            raise RuntimeError("Syntax error in form_key " + form_key)
        return numpy.array(stack.pop())

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
