from cachetools import LRUCache
from collections.abc import Mapping
import uproot4
import numpy
from coffea.nanoevents import transforms
from coffea.nanoevents.util import key_to_tuple, tuple_to_key


class UprootSourceMapping(Mapping):
    _debug = False

    def __init__(self, uuid_pfnmap, cache=None):
        self._uuid_pfnmap = uuid_pfnmap
        self._cache = cache
        self.setup()

    def setup(self):
        if self._cache is None:
            self._cache = LRUCache(10)
        self._uproot_options = {
            "array_cache": self._cache,
            "object_cache": self._cache,
        }

    def __getstate__(self):
        return {
            "uuid_pfnmap": self._uuid_pfnmap,
        }

    def __setstate__(self, state):
        self._uuid_pfnmap = state["uuid_pfnmap"]
        self._cache = None
        self.setup()

    def _tree(self, uuid, treepath):
        key = "UprootSourceMapping:" + tuple_to_key((uuid, treepath))
        try:
            return self._cache[key]
        except KeyError:
            pass
        pfn = self._uuid_pfnmap[uuid]
        tree = uproot4.open(pfn + ":" + treepath, **self._uproot_options)
        if str(tree.file.uuid) != uuid:
            raise RuntimeError(
                f"UUID of file {pfn} does not match expected value ({uuid})"
            )
        self._cache[key] = tree
        return tree

    def preload_tree(self, uuid, treepath, tree):
        """To save a double-open when using NanoEventsFactory.from_file"""
        key = "UprootSourceMapping:" + tuple_to_key((uuid, treepath))
        self._cache.update(tree.file.array_cache)
        self._cache.update(tree.file.object_cache)
        tree.file.array_cache = self._cache
        tree.file.object_cache = self._cache
        self._cache[key] = tree

    @classmethod
    def interpret_key(cls, key):
        uuid, treepath, entryrange, form_key, *layoutattr = key_to_tuple(key)
        start, stop = (int(x) for x in entryrange.split("-"))
        nodes = form_key.split(",")
        if len(layoutattr) == 1:
            nodes.append("!" + layoutattr[0])
        elif len(layoutattr) > 1:
            raise RuntimeError(f"Malformed key: {key}")
        return uuid, treepath, start, stop, nodes

    def __getitem__(self, key):
        uuid, treepath, start, stop, nodes = UprootSourceMapping.interpret_key(key)
        if UprootSourceMapping._debug:
            print("Gettting:", uuid, treepath, start, stop, nodes)
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
            raise RuntimeError(f"Syntax error in form key {nodes}")
        out = stack.pop()
        try:
            out = numpy.array(out)
        except ValueError:
            if UprootSourceMapping._debug:
                print(out)
            raise RuntimeError(
                f"Left with non-bare array after evaluating form key {nodes}"
            )
        return out

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class CachedMapping(Mapping):
    """A cache-wrapped mapping

    Reads will call into ``cache`` first, and if no key exists,
    the read will fall back to ``base``, saving the reult into ``cache``.
    """

    def __init__(self, cache, base):
        self.cache = cache
        self.base = base
        self.stats = {"hit": 0, "miss": 0}

    def __getitem__(self, key):
        try:
            value = self.cache[key]
            self.stats["hit"] += 1
            return value
        except KeyError:
            value = self.base[key]
            self.cache[key] = value
            self.stats["miss"] += 1
            return value

    def __iter__(self):
        return iter(self.base)

    def __len__(self):
        return len(self.base)
