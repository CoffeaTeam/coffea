import collections
import warnings
from ..util import awkward


class DataFrame(collections.abc.MutableMapping):
    """
    Simple delayed uproot reader (a la lazyarrays)
    Keeps track of values accessed, for later parsing.
    """
    def __init__(self, tree, stride=None, index=None, preload_items=None):
        self._tree = tree
        self._branchargs = {'awkwardlib': awkward}
        self._stride = None
        if (stride is not None) and (index is not None):
            self._stride = stride
            self._branchargs['entrystart'] = index*stride
            self._branchargs['entrystop'] = min(self._tree.numentries, (index+1)*stride)
        self._dict = {}
        self._materialized = set()
        if preload_items:
            self.preload(preload_items)

    def __delitem__(self, key):
        del self._dict[key]

    def __getitem__(self, key):
        if key in self._dict:
            return self._dict[key]
        elif key in self._tree:
            self._materialized.add(key)
            self._dict[key] = self._tree[key].array(**self._branchargs)
            return self._dict[key]
        else:
            raise KeyError(key)

    def __iter__(self):
        warnings.warning("An iterator has requested to read all branches from the tree", RuntimeWarning)
        for item in self._dict:
            self._materialized.add(item[0])
            yield item

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        self._dict[key] = value

    @property
    def available(self):
        return self._tree.keys()

    @property
    def materialized(self):
        return self._materialized

    @property
    def size(self):
        if self._stride is None:
            return self._tree.numentries
        return (self._branchargs['entrystop'] - self._branchargs['entrystart'])

    def preload(self, columns):
        for name in columns:
            if name in self._tree:
                _ = self[name]
