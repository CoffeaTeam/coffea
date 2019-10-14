from ..util import awkward

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping


class LazyDataFrame(MutableMapping):
    """Simple delayed uproot reader (a la lazyarrays)

    Keeps track of values accessed, for later parsing.

    Parameters
    ----------
        tree : uproot.TTree
            Tree to read
        stride : int, optional
            Size of chunk to read from the tree.
            Default: whole tree
        index : int, optional
            Chunk index to read
        preload_items : iterable
            Force preloading of a set of columns from the tree
        flatten : bool
            Remove jagged structure from columns read
    """
    def __init__(self, tree, stride=None, index=None, preload_items=None, flatten=False):
        self._tree = tree
        self._branchargs = {'awkwardlib': awkward, 'flatten': flatten}
        self._stride = None
        if (stride is not None) and (index is not None):
            self._stride = stride
            self._branchargs['entrystart'] = index * stride
            self._branchargs['entrystop'] = min(self._tree.numentries, (index + 1) * stride)
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

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(key)

    def __iter__(self):
        for item in self._dict:
            yield item

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        self._dict[key] = value

    @property
    def available(self):
        """List of available columns"""
        return self._tree.keys()

    @property
    def materialized(self):
        """List of columns read from tree"""
        return self._materialized

    @property
    def size(self):
        """Length of column vector"""
        if self._stride is None:
            return self._tree.numentries
        return (self._branchargs['entrystop'] - self._branchargs['entrystart'])

    def preload(self, columns):
        """Force loading of several columns

        Parameters
        ----------
            columns : iterable
                A list of columns to load
        """
        for name in columns:
            if name in self._tree:
                _ = self[name]


class PreloadedDataFrame(MutableMapping):
    """A dataframe for instances like spark where the columns are preloaded

    Provides a unified interface, matching that of LazyDataFrame.

    Parameters
    ----------
        size : int
            Number of rows
        items : dict
            Mapping of column name to column array
    """
    def __init__(self, size, items):
        self._size = size
        self._dict = items
        self._accessed = set()

    def __delitem__(self, key):
        del self._dict[key]

    def __getitem__(self, key):
        self._accessed.add(key)
        return self._dict[key]

    def __getattr__(self, key):
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(key)

    def __iter__(self):
        for key in self._dict:
            self._accessed.add(key)
            yield key

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        self._dict[key] = value

    @property
    def available(self):
        """List of available columns"""
        return self._dict.keys()

    @property
    def materialized(self):
        """List of accessed columns"""
        return self._accessed

    @property
    def size(self):
        """Length of column vector"""
        return self._size
