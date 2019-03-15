import collections
import warnings

class DataFrame(collections.abc.MutableMapping):
    """
    Simple delayed uproot reader (a la lazyarrays)
    Keeps track of values accessed, for later parsing.
    """
    def __init__(self, tree):
        self._tree = tree
        self._dict = {}
        self._materialized = set()

    def __delitem__(self, key):
        del self._dict[key]

    def __getitem__(self, key):
        if key in self._dict:
            return self._dict[key]
        elif key in self._tree:
            self._materialized.add(key)
            self._dict[key] = self._tree[key].array()
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
    def materialized(self):
        return self._materialized

    @property
    def size(self):
        return self._tree.numentries


