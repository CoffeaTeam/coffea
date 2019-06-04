from __future__ import division
from collections import namedtuple
from ..util import numpy as np
from ..processor.accumulator import AccumulatorABC
import copy
import functools
import math
import numbers
import re
import warnings

# Python 2 and 3 compatibility
_regex_pattern = re.compile("dummy").__class__
try:
    basestring
except NameError:
    basestring = str


MaybeSumSlice = namedtuple('MaybeSumSlice', ['start', 'stop', 'sum'])


def assemble_blocks(array, ndslice, depth=0):
    """
        Turns an n-dimensional slice of array (tuple of slices)
         into a nested list of numpy arrays that can be passed to np.block()

        Under the assumption that index 0 of any dimension is underflow, -2 overflow, -1 nanflow,
         this function will add the range not in the slice to the appropriate (over/under)flow bins
    """
    if depth == 0:
        ndslice = [MaybeSumSlice(s.start, s.stop, False) for s in ndslice]
    if depth == len(ndslice):
        slice_op = tuple(slice(s.start, s.stop) for s in ndslice)
        sum_op = tuple(i for i, s in enumerate(ndslice) if s.sum)
        return array[slice_op].sum(axis=sum_op, keepdims=True)
    slist = []
    newslice = ndslice[:]
    if ndslice[depth].start is not None:
        newslice[depth] = MaybeSumSlice(None, ndslice[depth].start, True)
        slist.append(assemble_blocks(array, newslice, depth + 1))
    newslice[depth] = MaybeSumSlice(ndslice[depth].start, ndslice[depth].stop, False)
    slist.append(assemble_blocks(array, newslice, depth + 1))
    if ndslice[depth].stop is not None:
        newslice[depth] = MaybeSumSlice(ndslice[depth].stop, -1, True)
        slist.append(assemble_blocks(array, newslice, depth + 1))
        newslice[depth] = MaybeSumSlice(-1, None, False)
        slist.append(assemble_blocks(array, newslice, depth + 1))
    return slist


def overflow_behavior(overflow):
    if overflow == 'none':
        return slice(1, -2)
    elif overflow == 'under':
        return slice(None, -2)
    elif overflow == 'over':
        return slice(1, -1)
    elif overflow == 'all':
        return slice(None, -1)
    elif overflow == 'allnan':
        return slice(None)
    elif overflow == 'justnan':
        return slice(-1, None)
    else:
        raise ValueError("Unrecognized overflow behavior: %s" % overflow)


@functools.total_ordering
class Interval(object):
    """
        Real number interval
        Totally ordered, assuming no overlap in intervals
        special nan interval is greater than [*, inf)
        string representation can be overriden by custom label
    """
    def __init__(self, lo, hi):
        self._lo = float(lo)
        self._hi = float(hi)
        self._label = None

    def __repr__(self):
        return "<%s (%s) instance at 0x%0x>" % (self.__class__.__name__, str(self), id(self))

    def __str__(self):
        if self._label is not None:
            return self._label
        if self.nan():
            return "(nanflow)"
        # string representation of floats is apparently a touchy subject.. further reading:
        # https://stackoverflow.com/questions/25898733/why-does-strfloat-return-more-digits-in-python-3-than-python-2
        return "%s%.12g, %.12g)" % ("(" if self._lo == -np.inf else "[", self._lo, self._hi)

    def __hash__(self):
        return hash(self._lo, self._hi)

    def __lt__(self, other):
        if other.nan() and not self.nan():
            return True
        elif self.nan():
            return False
        elif self._lo < other._lo:
            if self._hi > other._lo:
                raise ValueError("Intervals %r and %r intersect! What are you doing?!" % (self, other))
            return True
        return False

    def __eq__(self, other):
        if not isinstance(other, Interval):
            return False
        if other.nan() and self.nan():
            return True
        if self._lo == other._lo and self._hi == other._hi:
            return True
        return False

    def nan(self):
        return np.isnan(self._hi)

    @property
    def lo(self):
        return self._lo

    @property
    def hi(self):
        return self._hi

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, lbl):
        self._label = lbl


@functools.total_ordering
class StringBin(object):
    """
        A string used to fill a sparse axis
        Totally ordered, lexicographically by name
        The string representation can be overriden by custom label
    """
    def __init__(self, name, label=None):
        if not isinstance(name, basestring):
            raise TypeError("StringBin only supports string categories, received a %r" % name)
        elif '*' in name:
            raise ValueError("StringBin does not support character '*' as it conflicts with wildcard mapping.")
        self._name = name
        self._label = label

    def __repr__(self):
        return "<%s (%s) instance at 0x%0x>" % (self.__class__.__name__, self.name, id(self))

    def __str__(self):
        if self._label is not None:
            return self._label
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __lt__(self, other):
        return self._name < other._name

    def __eq__(self, other):
        if isinstance(other, StringBin):
            return self._name == other._name
        return False

    @property
    def name(self):
        return self._name

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, lbl):
        self._label = lbl


class Axis(object):
    """
        Axis: Base class for any type of axis
        Derived classes should implement, at least, an equality override
    """
    def __init__(self, name, label):
        if name == "weight":
            raise ValueError("Cannot create axis: 'weight' is a reserved keyword for Hist.fill()")
        self._name = name
        self._label = label

    def __repr__(self):
        return "<%s (name=%s) instance at 0x%0x>" % (self.__class__.__name__, self._name, id(self))

    @property
    def name(self):
        return self._name

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    def __eq__(self, other):
        if isinstance(other, Axis):
            if self._name != other._name:
                return False
            # label doesn't matter
            return True
        elif isinstance(other, basestring):
            # Convenient for testing axis in list by name
            if self._name != other:
                return False
            return True
        raise TypeError("Cannot compare an Axis with a %r" % other)


class SparseAxis(Axis):
    """
        SparseAxis: ABC for a sparse axis
        Derived should implement:
            index(identifier): return a hashable object for indexing
            __eq__(axis): axis has same definition (not necessarily same bins)
            __getitem__(index): return an identifier
            _ireduce(slice): return a list of hashes, slice is arbitrary

        What we really want here is a hashlist with some slice sugar on top
        It is usually the case that the identifier is already hashable,
          in which case index and __getitem__ are trivial, but this mechanism
          may be useful if the size of the tuple of identifiers in a
          sparse-binned histogram becomes too large
    """
    pass


class Cat(SparseAxis):
    """
        Specify a category axis with name and label
            name: is used as a keyword in histogram filling, immutable
            label: describes the meaning of the axis, can be changed
            sorting: axis sorting, 'identifier' or 'placement' order
        Number of categories is arbitrary, and filled sparsely
        Identifiers are strings
    """
    def __init__(self, name, label, sorting='identifier'):
        super(Cat, self).__init__(name, label)
        # In all cases key == value.name
        self._bins = {}
        self._sorting = sorting
        self._sorted = []

    def index(self, identifier):
        if isinstance(identifier, StringBin):
            index = identifier
        else:
            index = StringBin(identifier)
        if index.name not in self._bins:
            self._bins[index.name] = index
            self._sorted.append(index.name)
            if self._sorting == 'identifier':
                self._sorted.sort()
        return self._bins[index.name]

    def __eq__(self, other):
        # Sparse, so as long as name is the same
        return super(Cat, self).__eq__(other)

    def __getitem__(self, index):
        if not isinstance(index, StringBin):
            raise TypeError("Expected a StringBin object, got: %r" % index)
        identifier = index.name
        if identifier not in self._bins:
            raise KeyError("No identifier %r in this Category axis")
        return identifier

    def _ireduce(self, the_slice):
        out = None
        if isinstance(the_slice, StringBin):
            out = [the_slice.name]
        elif isinstance(the_slice, _regex_pattern):
            out = [k for k in self._sorted if the_slice.match(k)]
        elif isinstance(the_slice, basestring):
            pattern = "^" + re.escape(the_slice).replace(r'\*', '.*') + "$"
            m = re.compile(pattern)
            out = [k for k in self._sorted if m.match(k)]
        elif isinstance(the_slice, list):
            if not all(k in self._sorted for k in the_slice):
                warnings.warn("Not all requested indices present in %r" % self, RuntimeWarning)
            out = [k for k in self._sorted if k in the_slice]
        elif isinstance(the_slice, slice):
            if the_slice.step is not None:
                raise IndexError("Not sure how to use slice step for categories...")
            start, stop = 0, len(self._sorted)
            if isinstance(the_slice.start, basestring):
                start = self._sorted.index(the_slice.start)
            else:
                start = the_slice.start
            if isinstance(the_slice.stop, basestring):
                stop = self._sorted.index(the_slice.stop)
            else:
                stop = the_slice.stop
            out = self._sorted[start:stop]
        else:
            raise IndexError("Cannot understand slice %r on axis %r" % (the_slice, self))
        return [self._bins[k] for k in out]

    @property
    def size(self):
        return len(self._bins)

    def identifiers(self):
        return [self._bins[k] for k in self._sorted]


class DenseAxis(Axis):
    """
        DenseAxis: ABC for a fixed-size densely-indexed axis
        Derived should implement:
            index(identifier): return an index
            __eq__(axis): axis has same definition and binning
            __getitem__(index): return an identifier
            _ireduce(slice): return a slice or list of indices, input slice to be interpred as values
            reduced(islice): return a new axis with binning corresponding to the index slice (from _ireduce)
        TODO: hasoverflow(), not all dense axes might have an overflow concept, currently its implicitly assumed
            they do (as the only dense type is a numeric axis)
    """
    pass


class Bin(DenseAxis):
    """
        Specify a binned axis with name and label, and binning
            name: is used as a keyword in histogram filling, immutable
            label: describes the meaning of the axis, can be changed
            n_or_arr: number of bins, if uniform binning, otherwise a list (or numpy 1D array) of bin boundaries
            lo: if uniform binning, minimum value
            hi: if uniform binning, maximum value
        Axis will generate frequencies for n+3 bins, special bin indices:
            0 = underflow, n+1 = overflow, n+2 = nanflow
        Bin boundaries are [lo, hi)
    """
    def __init__(self, name, label, n_or_arr, lo=None, hi=None):
        super(Bin, self).__init__(name, label)
        if isinstance(n_or_arr, (list, np.ndarray)):
            self._uniform = False
            self._bins = np.array(n_or_arr, dtype='d')
            if not all(np.sort(self._bins) == self._bins):
                raise ValueError("Binning not sorted!")

            self._lo = self._bins[0]
            self._hi = self._bins[-1]
            # to make searchsorted differentiate inf from nan
            self._bins = np.append(self._bins, np.inf)
            interval_bins = np.r_[-np.inf, self._bins, np.nan]
            self._intervals = [Interval(low, high) for low, high in zip(interval_bins[:-1], interval_bins[1:])]
        elif isinstance(n_or_arr, numbers.Integral):
            if lo is None or hi is None:
                raise TypeError("Interpreting n_or_arr as uniform binning, please specify lo and hi values")
            self._uniform = True
            self._lo = lo
            self._hi = hi
            self._bins = n_or_arr
            interval_bins = np.r_[-np.inf, np.linspace(self._lo, self._hi, self._bins + 1), np.inf, np.nan]
            self._intervals = [Interval(low, high) for low, high in zip(interval_bins[:-1], interval_bins[1:])]
        else:
            raise TypeError("Cannot understand n_or_arr (nbins or binning array) type %r" % n_or_arr)

    def index(self, identifier):
        if (isinstance(identifier, np.ndarray) and len(identifier.shape) == 1) or isinstance(identifier, numbers.Number):
            if self._uniform:
                idx = np.clip(np.floor((identifier - self._lo) * self._bins / (self._hi - self._lo)) + 1, 0, self._bins + 1)
                if isinstance(idx, np.ndarray):
                    idx[np.isnan(idx)] = self.size - 1
                    idx = idx.astype(int)
                elif np.isnan(idx):
                    idx = self.size - 1
                else:
                    idx = int(idx)
                return idx
            else:
                return np.searchsorted(self._bins, identifier, side='right')
        elif isinstance(identifier, Interval):
            if identifier.nan():
                return self.size - 1
            return self.index(identifier._lo)
        raise TypeError("Request bin indices with a identifier or 1-D array only")

    def __eq__(self, other):
        if isinstance(other, DenseAxis):
            if not super(Bin, self).__eq__(other):
                return False
            if self._uniform != other._uniform:
                return False
            if self._uniform and self._bins != other._bins:
                return False
            if not self._uniform and not all(self._bins == other._bins):
                return False
            return True
        return super(Bin, self).__eq__(other)

    def __getitem__(self, index):
        return self._intervals[index]

    def _ireduce(self, the_slice):
        if isinstance(the_slice, numbers.Number):
            the_slice = slice(the_slice, the_slice)
        elif isinstance(the_slice, Interval):
            if the_slice.nan():
                return slice(-1, None)
            lo = the_slice._lo if the_slice._lo > -np.inf else None
            hi = the_slice._hi if the_slice._hi < np.inf else None
            the_slice = slice(lo, hi)
        if isinstance(the_slice, slice):
            blo, bhi = None, None
            if the_slice.start is not None:
                if the_slice.start < self._lo:
                    raise ValueError("Reducing along axis %r: requested start %r exceeds bin boundaries (use open slicing, e.g. x[:stop])" % (self,
                                                                                                                                              the_slice.start))
                if self._uniform:
                    blo_real = (the_slice.start - self._lo) * self._bins / (self._hi - self._lo) + 1
                    blo = np.clip(np.round(blo_real).astype(int), 0, self._bins + 1)
                    if abs(blo - blo_real) > 1.e-14:
                        warnings.warn("Reducing along axis %r: requested start %r between bin boundaries, no interpolation is performed" % (self,
                                                                                                                                            the_slice.start),
                                      RuntimeWarning)
                else:
                    if the_slice.start not in self._bins:
                        warnings.warn("Reducing along axis %r: requested start %r between bin boundaries, no interpolation is performed" % (self,
                                                                                                                                            the_slice.start),
                                      RuntimeWarning)
                    blo = self.index(the_slice.start)
            if the_slice.stop is not None:
                if the_slice.stop > self._hi:
                    raise ValueError("Reducing along axis %r: requested stop %r exceeds bin boundaries (use open slicing, e.g. x[start:])" % (self,
                                                                                                                                              the_slice.stop))
                if self._uniform:
                    bhi_real = (the_slice.stop - self._lo) * self._bins / (self._hi - self._lo) + 1
                    bhi = np.clip(np.round(bhi_real).astype(int), 0, self._bins + 1)
                    if abs(bhi - bhi_real) > 1.e-14:
                        warnings.warn("Reducing along axis %r: requested stop %r between bin boundaries, no interpolation is performed" % (self,
                                                                                                                                           the_slice.stop),
                                      RuntimeWarning)
                else:
                    if the_slice.stop not in self._bins:
                        warnings.warn("Reducing along axis %r: requested stop %r between bin boundaries, no interpolation is performed" % (self,
                                                                                                                                           the_slice.stop),
                                      RuntimeWarning)
                    bhi = self.index(the_slice.stop)
                # Assume null ranges (start==stop) mean we want the bin containing the value
                if blo is not None and blo == bhi:
                    bhi += 1
            if the_slice.step is not None:
                raise NotImplementedError("Step slicing can be interpreted as a rebin factor")
            return slice(blo, bhi, the_slice.step)
        elif isinstance(the_slice, list) and all(isinstance(v, Interval) for v in the_slice):
            raise NotImplementedError("Slice histogram from list of intervals")
        raise IndexError("Cannot understand slice %r on axis %r" % (the_slice, self))

    def reduced(self, islice):
        """
            Return a new axis with binning corresponding to the slice made on this axis
            overflow will be taken care of by Hist.__getitem__
            islice should be as returned from _ireduce, start and stop should be None or within [1, size()-1]
        """
        if islice.step is not None:
            raise NotImplementedError("Step slicing can be interpreted as a rebin factor")
        if islice.start is None and islice.stop is None:
            return self
        if self._uniform:
            lo = self._lo
            ilo = 0
            if islice.start is not None:
                lo += (islice.start - 1) * (self._hi - self._lo) / self._bins
                ilo = islice.start - 1
            hi = self._hi
            ihi = self._bins
            if islice.stop is not None:
                hi = self._lo + (islice.stop - 1) * (self._hi - self._lo) / self._bins
                ihi = islice.stop - 1
            bins = ihi - ilo
            # TODO: remove this once satisfied it works
            rbins = (hi - lo) * self._bins / (self._hi - self._lo)
            assert abs(bins - rbins) < 1e-14, "%d %f %r" % (bins, rbins, self)
            ax = Bin(self._name, self._label, bins, lo, hi)
            return ax
        else:
            lo = None if islice.start is None else islice.start - 1
            hi = -1 if islice.stop is None else islice.stop
            bins = self._bins[slice(lo, hi)]
            ax = Bin(self._name, self._label, bins)
            return ax

    @property
    def size(self):
        if self._uniform:
            return self._bins + 3
        # (inf added at constructor)
        return len(self._bins) + 1

    def edges(self, overflow='none'):
        """
            Bin boundaries
                overflow: create overflow and/or underflow bins by adding a bin of same width to each end
                    only 'none', 'under', 'over', 'all' types are supported
        """
        if self._uniform:
            out = np.linspace(self._lo, self._hi, self._bins + 1)
        else:
            out = self._bins[:-1].copy()
        out = np.r_[2 * out[0] - out[1], out, 2 * out[-1] - out[-2], 3 * out[-1] - 2 * out[-2]]
        return out[overflow_behavior(overflow)]

    def centers(self, overflow='none'):
        edges = self.edges(overflow)
        return (edges[:-1] + edges[1:]) / 2

    def identifiers(self, overflow='none'):
        return self._intervals[overflow_behavior(overflow)]


class Hist(AccumulatorABC):
    """
        Specify a multidimensional histogram
            label: description of meaning of frequencies (axis descriptions specified in axis constructor)
            dtype: underlying numpy dtype of frequencies
            *axes: positional list of Cat or Bin objects
    """
    DEFAULT_DTYPE = 'd'

    def __init__(self, label, *axes, **kwargs):
        if not isinstance(label, basestring):
            raise TypeError("label must be a string")
        self._label = label
        self._dtype = kwargs.pop('dtype', Hist.DEFAULT_DTYPE)  # Much nicer in python3 :(
        if not all(isinstance(ax, Axis) for ax in axes):
            raise TypeError("All axes must be derived from Axis class")
        # if we stably partition axes to sparse, then dense, some things simplify
        # ..but then the user would then see the order change under them
        self._axes = axes
        self._dense_shape = tuple([ax.size for ax in self._axes if isinstance(ax, DenseAxis)])
        if np.prod(self._dense_shape) > 10000000:
            warnings.warn("Allocating a large (>10M bin) histogram!", RuntimeWarning)
        self._sumw = {}
        # Storage of sumw2 starts at first use of weight keyword in fill()
        self._sumw2 = None

    def __repr__(self):
        return "<%s (%s) instance at 0x%0x>" % (self.__class__.__name__, ",".join(d.name for d in self.axes()), id(self))

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    def copy(self, content=True):
        out = Hist(self._label, *self._axes, dtype=self._dtype)
        if self._sumw2 is not None:
            out._sumw2 = {}
        if content:
            out._sumw = copy.deepcopy(self._sumw)
            out._sumw2 = copy.deepcopy(self._sumw2)
        return out

    def identity(self):
        return self.copy(content=False)

    def clear(self):
        self._sumw = {}
        self._sumw2 = None

    def axis(self, axis_name):
        if axis_name in self._axes:
            return self._axes[self._axes.index(axis_name)]
        raise KeyError("No axis %s found in %r" % (axis_name, self))

    def axes(self):
        return self._axes

    @property
    def fields(self):
        """
            Stub for histbook compatibility in striped
        """
        return [ax.name for ax in self._axes]

    def dim(self):
        return len(self._axes)

    def dense_dim(self):
        return len(self._dense_shape)

    def sparse_dim(self):
        return self.dim() - self.dense_dim()

    def dense_axes(self):
        return [ax for ax in self._axes if isinstance(ax, DenseAxis)]

    def sparse_axes(self):
        return [ax for ax in self._axes if isinstance(ax, SparseAxis)]

    def sparse_nbins(self):
        return len(self._sumw)

    def _idense(self, axis):
        return self.dense_axes().index(axis)

    def _isparse(self, axis):
        return self.sparse_axes().index(axis)

    def _init_sumw2(self):
        self._sumw2 = {}
        for key in self._sumw.keys():
            self._sumw2[key] = self._sumw[key].copy()

    def compatible(self, other):
        """
            Checks if this histogram is compatible with another, i.e. they have identical binning
        """
        if self.dim() != other.dim():
            return False
        if set(d.name for d in self.sparse_axes()) != set(d.name for d in other.sparse_axes()):
            return False
        if not all(d1 == d2 for d1, d2 in zip(self.dense_axes(), other.dense_axes())):
            return False
        return True

    def add(self, other):
        if not self.compatible(other):
            raise ValueError("Cannot add this histogram with histogram %r of dissimilar dimensions" % other)

        raxes = other.sparse_axes()

        def add_dict(l, r):
            for rkey in r.keys():
                lkey = tuple(self.axis(rax).index(rax[ridx]) for rax, ridx in zip(raxes, rkey))
                if lkey in l:
                    l[lkey] += r[rkey]
                else:
                    l[lkey] = copy.deepcopy(r[rkey])

        if self._sumw2 is None and other._sumw2 is None:
            pass
        elif self._sumw2 is None:
            self._init_sumw2()
            add_dict(self._sumw2, other._sumw2)
        elif other._sumw2 is None:
            add_dict(self._sumw2, other._sumw)
        else:
            add_dict(self._sumw2, other._sumw2)
        add_dict(self._sumw, other._sumw)
        return self

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) > self.dim():
            raise IndexError("Too many indices for this histogram")
        elif len(keys) < self.dim():
            if Ellipsis in keys:
                idx = keys.index(Ellipsis)
                slices = (slice(None),) * (self.dim() - len(keys) + 1)
                keys = keys[:idx] + slices + keys[idx + 1:]
            else:
                slices = (slice(None),) * (self.dim() - len(keys))
                keys += slices
        sparse_idx = []
        dense_idx = []
        new_dims = []
        for s, ax in zip(keys, self._axes):
            if isinstance(ax, SparseAxis):
                sparse_idx.append(ax._ireduce(s))
                new_dims.append(ax)
            else:
                islice = ax._ireduce(s)
                dense_idx.append(islice)
                new_dims.append(ax.reduced(islice))
        dense_idx = tuple(dense_idx)

        def dense_op(array):
            return np.block(assemble_blocks(array, dense_idx))

        out = Hist(self._label, *new_dims, dtype=self._dtype)
        if self._sumw2 is not None:
            out._init_sumw2()
        for sparse_key in self._sumw:
            if not all(k in idx for k, idx in zip(sparse_key, sparse_idx)):
                continue
            if sparse_key in out._sumw:
                out._sumw[sparse_key] += dense_op(self._sumw[sparse_key])
                if self._sumw2 is not None:
                    out._sumw2[sparse_key] += dense_op(self._sumw2[sparse_key])
            else:
                out._sumw[sparse_key] = dense_op(self._sumw[sparse_key]).copy()
                if self._sumw2 is not None:
                    out._sumw2[sparse_key] = dense_op(self._sumw2[sparse_key]).copy()
        return out

    def fill(self, **values):
        if not all(d.name in values for d in self._axes):
            missing = ", ".join(d.name for d in self._axes if d.name not in values)
            raise ValueError("Not all axes specified for %r.  Missing: %s" % (self, missing))

        if "weight" in values and self._sumw2 is None:
            self._init_sumw2()

        sparse_key = tuple(d.index(values[d.name]) for d in self.sparse_axes())
        if sparse_key not in self._sumw:
            self._sumw[sparse_key] = np.zeros(shape=self._dense_shape, dtype=self._dtype)
            if self._sumw2 is not None:
                self._sumw2[sparse_key] = np.zeros(shape=self._dense_shape, dtype=self._dtype)

        if self.dense_dim() > 0:
            dense_indices = tuple(d.index(values[d.name]) for d in self._axes if isinstance(d, DenseAxis))
            if "weight" in values:
                np.add.at(self._sumw[sparse_key], dense_indices, values["weight"])
                np.add.at(self._sumw2[sparse_key], dense_indices, values["weight"]**2)
            else:
                np.add.at(self._sumw[sparse_key], dense_indices, 1.)
                if self._sumw2 is not None:
                    np.add.at(self._sumw2[sparse_key], dense_indices, 1.)
        else:
            if "weight" in values:
                self._sumw[sparse_key] += np.sum(values["weight"])
                self._sumw2[sparse_key] += np.sum(values["weight"]**2)
            else:
                self._sumw[sparse_key] += 1.
                if self._sumw2 is not None:
                    self._sumw2[sparse_key] += 1.

    def sum(self, *axes, **kwargs):
        """
            Integrates out a set of axes, producing a new histogram
                *axes: axes to integrate out (either name or Axis object)
                overflow: 'none', 'under', 'over', 'all', 'allnan' (only applies to dense axes)
        """
        overflow = kwargs.pop('overflow', 'none')
        axes = [self.axis(ax) for ax in axes]
        reduced_dims = [ax for ax in self._axes if ax not in axes]
        out = Hist(self._label, *reduced_dims, dtype=self._dtype)
        if self._sumw2 is not None:
            out._init_sumw2()

        sparse_drop = []
        dense_slice = [slice(None)] * self.dense_dim()
        dense_sum_dim = []
        for axis in axes:
            if isinstance(axis, DenseAxis):
                idense = self._idense(axis)
                dense_sum_dim.append(idense)
                dense_slice[idense] = overflow_behavior(overflow)
            elif isinstance(axis, SparseAxis):
                isparse = self._isparse(axis)
                sparse_drop.append(isparse)
        dense_slice = tuple(dense_slice)
        dense_sum_dim = tuple(dense_sum_dim)

        def dense_op(array):
            if len(dense_sum_dim) > 0:
                return np.sum(array[dense_slice], axis=dense_sum_dim)
            return array

        for key in self._sumw.keys():
            new_key = tuple(k for i, k in enumerate(key) if i not in sparse_drop)
            if new_key in out._sumw:
                out._sumw[new_key] += dense_op(self._sumw[key])
                if self._sumw2 is not None:
                    out._sumw2[new_key] += dense_op(self._sumw2[key])
            else:
                out._sumw[new_key] = dense_op(self._sumw[key]).copy()
                if self._sumw2 is not None:
                    out._sumw2[new_key] = dense_op(self._sumw2[key]).copy()
        return out

    def project(self, axis_name, the_slice=slice(None), overflow='none'):
        """
            Projects current histogram down one dimension
                axis_name: dimension to reduce on
                the_slice: any slice, list, string, or other object that the axis will understand
                overflow: see sum() description for allowed values
            N.B. the more idiomatic way is to slice and sum, although this may be more readable
        """
        axis = self.axis(axis_name)
        full_slice = tuple(slice(None) if ax != axis else the_slice for ax in self._axes)
        if isinstance(the_slice, Interval):
            # Handle overflow intervals nicely
            if the_slice.nan():
                overflow = 'justnan'
            elif the_slice.lo == -np.inf:
                overflow = 'under'
            elif the_slice.hi == np.inf:
                overflow = 'over'
        return self[full_slice].sum(axis.name, overflow=overflow)  # slice may make new axis, use name

    def profile(self, axis_name):
        raise NotImplementedError("Profiling along an axis")

    def group(self, new_axis, old_axes, mapping, overflow='none'):
        """
            Group a set of slices on old axes into a single new axis
                new_axis: A new sparse dimension
                old_axes: axis or tuple of axes which are being grouped
                mapping: dictionary of {'new_bin': (slice, ...), ...}
                    where each slice is on the axes being re-binned
                overflow: see sum() description for allowed values
        """
        if not isinstance(new_axis, SparseAxis):
            raise TypeError("New axis must be a sparse axis")
        if not isinstance(old_axes, tuple):
            old_axes = (old_axes,)
        old_axes = [self.axis(ax) for ax in old_axes]
        old_indices = [i for i, ax in enumerate(self._axes) if ax in old_axes]
        new_dims = [new_axis] + [ax for ax in self._axes if ax not in old_axes]
        out = Hist(self._label, *new_dims, dtype=self._dtype)
        if self._sumw2 is not None:
            out._init_sumw2()
        for new_cat in mapping.keys():
            the_slice = mapping[new_cat]
            if not isinstance(the_slice, tuple):
                the_slice = (the_slice,)
            if len(the_slice) != len(old_axes):
                raise Exception("Slicing does not match number of axes being rebinned")
            full_slice = [slice(None)] * self.dim()
            for idx, s in zip(old_indices, the_slice):
                full_slice[idx] = s
            full_slice = tuple(full_slice)
            reduced_hist = self[full_slice].sum(*tuple(ax.name for ax in old_axes), overflow=overflow)  # slice may change old axis binning
            new_idx = new_axis.index(new_cat)
            for key in reduced_hist._sumw:
                new_key = (new_idx,) + key
                out._sumw[new_key] = reduced_hist._sumw[key]
                if self._sumw2 is not None:
                    out._sumw2[new_key] = reduced_hist._sumw2[key]
        return out

    def rebin(self, old_axis, new_axis):
        """
            Rebin a dense axis
                old_axis: name or Axis object to rebin
                new_axis: Dense Axis object defining new axis
            This function will construct the mapping from old to new axis
        """
        old_axis = self.axis(old_axis)
        new_dims = [ax if ax != old_axis else new_axis for ax in self._axes]
        out = Hist(self._label, *new_dims, dtype=self._dtype)
        if self._sumw2 is not None:
            out._init_sumw2()

        # would have been nice to use ufunc.reduceat, but we should support arbitrary reshuffling
        idense = self._idense(old_axis)

        def view_ax(idx):
            fullindex = [slice(None)] * self.dense_dim()
            fullindex[idense] = idx
            return tuple(fullindex)
        binmap = [new_axis.index(i) for i in old_axis.identifiers(overflow='allnan')]

        def dense_op(array):
            anew = np.zeros(out._dense_shape, dtype=out._dtype)
            for iold, inew in enumerate(binmap):
                anew[view_ax(inew)] += array[view_ax(iold)]
            return anew

        for key in self._sumw:
            out._sumw[key] = dense_op(self._sumw[key])
            if self._sumw2 is not None:
                out._sumw2[key] = dense_op(self._sumw2[key])
        return out

    def values(self, sumw2=False, overflow='none'):
        """
            Returns dict of (sparse identifier, ...): np.array(frequencies)
            sumw2: if True, frequencies is a tuple (sum weights, sum sqaured weights)
            overflow: see sum() description for allowed values
        """
        def view_dim(arr):
            if self.dense_dim() == 0:
                return arr
            else:
                return arr[tuple(overflow_behavior(overflow) for _ in range(self.dense_dim()))]

        out = {}
        for sparse_key in self._sumw.keys():
            id_key = tuple(ax[k] for ax, k in zip(self.sparse_axes(), sparse_key))
            if sumw2:
                if self._sumw2 is not None:
                    w2 = view_dim(self._sumw2[sparse_key])
                else:
                    w2 = view_dim(self._sumw[sparse_key])
                out[id_key] = (view_dim(self._sumw[sparse_key]), w2)
            else:
                out[id_key] = view_dim(self._sumw[sparse_key])
        return out

    def scale(self, factor, axis=None):
        """
            Scale histogram in-place by factor
                factor: number of dict of numbers
                axis: which (sparse) axis the dict applies to
        """
        if self._sumw2 is None:
            self._init_sumw2()
        if isinstance(factor, numbers.Number) and axis is None:
            for key in self._sumw.keys():
                self._sumw[key] *= factor
                self._sumw2[key] *= factor**2
        elif isinstance(factor, dict):
            axis = self.axis(axis)
            isparse = self._isparse(axis)
            factor = dict((axis.index(k), v) for k, v in factor.items())
            for key in self._sumw.keys():
                if key[isparse] in factor:
                    self._sumw[key] *= factor[key[isparse]]
                    self._sumw2[key] *= factor[key[isparse]]**2
        elif isinstance(factor, np.ndarray):
            axis = self.axis(axis)
            raise NotImplementedError("Scale dense dimension by a factor")
        else:
            raise TypeError("Could not interpret scale factor")

    def identifiers(self, axis, overflow='none'):
        """
            Return identifiers of axis which appear in histogram.
                axis: name or Axis object
                overflow: see sum() description
        """
        axis = self.axis(axis)
        if isinstance(axis, SparseAxis):
            out = []
            isparse = self._isparse(axis)
            for identifier in axis.identifiers():
                if any(k[isparse] == axis.index(identifier) for k in self._sumw.keys()):
                    out.append(identifier)
            return out
        elif isinstance(axis, DenseAxis):
            return axis.identifiers(overflow=overflow)
