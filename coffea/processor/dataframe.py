from collections.abc import MutableMapping
import awkward1
import uproot4
from coffea.util import deprecate_detected_awkward0


class LazyDataFrame(MutableMapping):
    """Simple delayed uproot reader (a la lazyarrays)

    One can access branches either through ``df["bname"]`` or ``df.bname``, although
    the latter is restricted to branches that do not start with a leading underscore.
    Keeps track of values accessed, in the `materialized` attribute.

    Parameters
    ----------
        tree : uproot4.TTree
            Tree to read
        entrystart : int, optional
            First entry to read, default: 0
        entrystop : int, optional
            Last entry to read, default None (read to end)
        preload_items : iterable
            Force preloading of a set of columns from the tree
        flatten : bool
            Remove jagged structure from columns read
    """

    def __init__(
        self, tree, entrystart=None, entrystop=None, preload_items=None, flatten=False
    ):
        self._tree = tree
        self._flatten = flatten
        self._branchargs = {
            "decompression_executor": uproot4.source.futures.TrivialExecutor(),
            "interpretation_executor": uproot4.source.futures.TrivialExecutor(),
        }
        if entrystart is None or entrystart < 0:
            entrystart = 0
        if entrystop is None or entrystop > tree.num_entries:
            entrystop = tree.num_entries
        self._branchargs["entry_start"] = entrystart
        self._branchargs["entry_stop"] = entrystop
        self._available = {k for k in self._tree.keys()}
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
            array = self._tree[key].array(**self._branchargs)
            if self._flatten and isinstance(awkward1.type(array).type, awkward1.types.ListType):
                array = awkward1.flatten(array)
            array = awkward1.to_awkward0(array)
            deprecate_detected_awkward0(array)
            self._dict[key] = array
            return self._dict[key]
        else:
            raise KeyError(key)

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(key)

    def __iter__(self):
        for item in self._available:
            yield item

    def __len__(self):
        return len(self._dict)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __contains__(self, key):
        # by default, MutableMapping uses __getitem__ to test, but we want to avoid materialization
        return key in self._dict or key in self._tree

    @property
    def available(self):
        """Set of available columns"""
        return self._available

    @property
    def columns(self):
        """Set of available columns"""
        return self._available

    @property
    def materialized(self):
        """Set of columns read from tree"""
        return self._materialized

    @property
    def size(self):
        """Length of column vector"""
        return self._branchargs["entry_stop"] - self._branchargs["entry_start"]

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
        out = self._dict[key]
        deprecate_detected_awkward0(out)
        return out

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self.__getitem__(key)
        except KeyError:
            raise AttributeError(key)

    def __iter__(self):
        for key in self._dict:
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
