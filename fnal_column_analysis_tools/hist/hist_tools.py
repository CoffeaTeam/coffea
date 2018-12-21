from __future__ import division
import numbers
import math
import warnings
import copy
import re
import numpy as np

# Different in python 2 and 3
_regex_pattern = re.compile("dummy").__class__

class Axis(object):
    # TODO: ABC? All derived must implement index(scalar), size(), equality
    def __init__(self, name, label):
        if name == "weight":
            raise ValueError("Cannot create axis: 'weight' is a reserved keyword for histograms")
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
        elif isinstance(other, str):
            # Convenient for testing axis in list by name
            if self._name != other:
                return False
            return True
        raise TypeError("Cannot compare an Axis with a %r" % other)


class Cat(Axis):
    """
        Specify a category axis with name and label
            name: is used as a keyword in histogram filling, immutable
            label: describes the meaning of the axis, can be changed
        Number of categories is arbitrary, and filled sparsely
    """
    def __init__(self, name, label):
        super(Cat, self).__init__(name, label)
        # TODO: SortedList from sortedcontainers ?
        self._categories = []

    def index(self, scalar):
        if not isinstance(scalar, str):
            raise TypeError("Cat axis supports only string categories")
        # TODO: do we need some sort of hashing or just go by string?
        if scalar not in self._categories:
            self._categories.append(scalar)
        return scalar

    @property
    def size(self):
        return len(self._categories)

    def __eq__(self, other):
        # Sparse, so as long as name is the same
        return super(Cat, self).__eq__(other)

    def _ireduce(self, the_slice):
        self._categories.sort()
        if isinstance(the_slice, _regex_pattern):
            return [v for v in self._categories if the_slice.match(v)]
        elif isinstance(the_slice, str):
            pattern = "^" + the_slice.replace('*', '.*')
            m = re.compile(pattern)
            return [v for v in self._categories if m.match(v)]
        elif isinstance(the_slice, list):
            if not all(v in self._categories for v in the_slice):
                raise KeyError("Not all requested indices present in %r" % self)
            return the_slice
        elif isinstance(the_slice, slice):
            if the_slice.step is not None:
                raise IndexError("Not sure how to use slice step for categories...")
            start, stop = 0, len(self._categories)
            if isinstance(the_slice.start, str):
                start = self._categories.index(the_slice.start)
            if isinstance(the_slice.stop, str):
                stop = self._categories.index(the_slice.stop)
            return self._categories[start:stop]
        raise IndexError("Cannot understand slice %r on axis %r" % (the_slice, self))


class Bin(Axis):
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
            if not all(np.sort(self._bins)==self._bins):
                raise ValueError("Binning not sorted!")

            self._lo = self._bins[0]
            self._hi = self._bins[-1]
            # to make searchsorted differentiate inf from nan
            np.append(self._bins, np.inf)
        elif isinstance(n_or_arr, numbers.Integral):
            if lo is None or hi is None:
                raise TypeError("Interpreting n_or_arr as uniform binning, please specify lo and hi values")
            self._uniform = True
            self._lo = lo
            self._hi = hi
            self._bins = n_or_arr
        else:
            raise TypeError("Cannot understand n_or_arr (nbins or binning array) type %r" % n_or_arr)

    def index(self, scalar):
        if (isinstance(scalar, np.ndarray) and len(scalar.shape)==1) or isinstance(scalar, numbers.Number):
            if self._uniform:
                idx = np.clip(np.floor((scalar-self._lo)*self._bins/(self._hi-self._lo)) + 1, 0, self._bins+1)
                if isinstance(idx, np.ndarray):
                    idx[idx==np.nan] = self.size-1
                    idx = idx.astype(int)
                elif np.isnan(idx):
                    idx = self.size-1
                else:
                    idx = int(idx)
                return idx
            else:
                return np.searchsorted(self._bins, scalar, side='right')
        raise TypeError("Request bin indices with a scalar or 1-D array only")

    @property
    def size(self):
        if self._uniform:
            return self._bins + 3
        # (inf added at constructor)
        return len(self._bins)+2

    def __eq__(self, other):
        if isinstance(other, Bin):
            if not super(Bin, self).__eq__(other):
                return False
            if self._uniform != other._uniform:
                return False
            if not ((self._uniform and self._bins==other._bins) or all(self._bins==other._bins)):
                return False
            return True
        return super(Bin, self).__eq__(other)

    def _ireduce(self, the_slice):
        if isinstance(the_slice, numbers.Number):
            the_slice = slice(the_slice, the_slice)
        if isinstance(the_slice, slice):
            blo, bhi = None, None
            if the_slice.start is not None:
                if self._uniform:
                    blo = self.index(the_slice.start)
                    blo_ceil = np.clip(np.ceil((the_slice.start-self._lo)*self._bins/(self._hi-self._lo)) + 1, 0, self._bins+1)
                    if blo == 0:
                        warnings.warn("Reducing along axis %r: requested start %r exceeds bin boundaries" % (self, the_slice.start), RuntimeWarning)
                    elif blo_ceil != blo:
                        warnings.warn("Reducing along axis %r: requested start %r between bin boundaries, no interpolation is performed" % (self, the_slice.start), RuntimeWarning)
                else:
                    if the_slice.start not in self._bins:
                        warnings.warn("Reducing along axis %r: requested start %r between bin boundaries, no interpolation is performed" % (self, the_slice.start), RuntimeWarning)
                    blo = self.index(the_slice.start)
            if the_slice.stop is not None:
                if self._uniform:
                    bhi = self.index(the_slice.stop)
                    bhi_ceil = np.clip(np.ceil((the_slice.stop-self._lo)*self._bins/(self._hi-self._lo)) + 1, 0, self._bins+1)
                    if bhi >= self.size-2:
                        warnings.warn("Reducing along axis %r: requested stop %r exceeds bin boundaries" % (self, the_slice.stop), RuntimeWarning)
                    elif bhi_ceil != bhi:
                        warnings.warn("Reducing along axis %r: requested stop %r between bin boundaries, no interpolation is performed" % (self, the_slice.stop), RuntimeWarning)
                else:
                    if the_slice.stop not in self._bins:
                        warnings.warn("Reducing along axis %r: requested stop %r between bin boundaries, no interpolation is performed" % (self, the_slice.stop), RuntimeWarning)
                    bhi = self.index(the_slice.stop)
                # Assume null ranges (start==stop) mean we want the bin containing the value
                if blo is not None and blo == bhi:
                    bhi += 1
            return slice(blo, bhi, the_slice.step)
        raise IndexError("Cannot understand slice %r on axis %r" % (the_slice, self))

    def edges(self, extended=False):
        """
            Bin boundaries
                extended: create overflow and underflow bins by adding a bin of same width to each end
        """
        if self._uniform:
            out = np.linspace(self._lo, self._hi, self._bins+1)
        else:
            out = self._bins.copy()
        if extended:
            out = np.r_[2*out[0]-out[1], out, 2*out[-1]-out[-2]]
        return out

    def centers(self, extended=False):
        edges = self.edges(extended)
        return (edges[:-1]+edges[1:])/2


class Hist(object):
    """
        Specify a multidimensional histogram
            label: description of meaning of frequencies (axis descriptions specified in axis constructor)
            dtype: underlying numpy dtype of frequencies
            *axes: positional list of Cat or Bin objects
    """
    def __init__(self, label, *axes, **kwargs):
        if not isinstance(label, str):
            raise TypeError("label must be a string")
        self._label = label
        self._dtype = kwargs.pop('dtype', 'd')  # Much nicer in python3 :(
        if not all(isinstance(ax, Axis) for ax in axes):
            raise TypeError("All axes must be derived from Axis class")
        # if we stably partition axes to sparse, then dense, some things simplify
        # ..but then the user would then see the order change under them
        self._axes = axes
        self._dense_shape = tuple([ax.size for ax in self._axes if isinstance(ax, Bin)])
        if np.prod(self._dense_shape) > 1000000:
            warnings.warn("Allocating a large (>1M bin) histogram!", RuntimeWarning)
        self._sumw = {}
        # Storage of sumw2 starts at first use of weight keyword in fill()
        self._sumw2 = None

    def __repr__(self):
        return "<%s (%s) instance at 0x%0x>" % (self.__class__.__name__, ",".join(d.name for d in self.axes()), id(self))

    def copy(self, content=True):
        out = Hist(self._label, *self._axes, dtype=self._dtype)
        if self._sumw2 is not None:
            out._sumw2 = {}
        if content:
            out._sumw = copy.deepcopy(self._sumw)
            out._sumw2 = copy.deepcopy(self._sumw2)
        return out

    def clear(self):
        self._sumw = {}
        self._sumw2 = None

    def axis(self, axis_name):
        if axis_name in self._axes:
            return self._axes[self._axes.index(axis_name)]
        raise KeyError("No axis named %s found in %r" % (axis_name, self))

    def axes(self):
        return self._axes

    def dense_dim(self):
        return len(self._dense_shape)

    def sparse_dim(self):
        return len(self._axes) - self.dense_dim()

    def dense_axes(self):
        return [ax for ax in self._axes if isinstance(ax, Bin)]

    def sparse_axes(self):
        return [ax for ax in self._axes if isinstance(ax, Cat)]

    def _idense(self, axis):
        return self.dense_axes().index(axis)

    def _isparse(self, axis):
        return self.sparse_axes().index(axis)

    def _init_sumw2(self):
        self._sumw2 = {}
        for key in self._sumw.keys():
            self._sumw2[key] = self._sumw[key].copy()

    def __iadd__(self, other):
        if len(self._axes)!=len(other._axes):
            raise ValueError("Cannot add this histogram with histogram %r of dissimilar dimensions" % other)
        if set(d.name for d in self._axes) != set(d.name for d in other._axes):
            raise ValueError("Cannot add this histogram with histogram %r of dissimilar dimensions" % other)

        raxes = other.sparse_axes()
        def add_dict(l, r):
            for rkey in r.keys():
                lkey = tuple(self.axis(rax).index(rval) for rax, rval in zip(raxes, rkey))
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

    def __add__(self, other):
        out = self.copy()
        out += other
        return out

    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) > len(self._axes):
            raise IndexError("Too many indices for this histogram")
        elif len(keys) < len(self._axes):
            if Ellipsis in keys:
                idx = keys.index(Ellipsis)
                slices = (slice(None),)*(len(self._axes)-len(keys)+1)
                keys = keys[:idx] + slices + keys[idx+1:]
            else:
                slices = (slice(None),)*(len(self._axes)-len(keys))
                keys += slices
        sparse_idx = []
        dense_idx = []
        for s,ax in zip(keys, self._axes):
            if isinstance(ax, Cat):
                sparse_idx.append(ax._ireduce(s))
            else:
                dense_idx.append(ax._ireduce(s))
        dense_idx = tuple(dense_idx)

        out = self.copy(content=False)
        for sparse_key in self._sumw:
            if not all(k in idx for k,idx in zip(sparse_key, sparse_idx)):
                continue
            if sparse_key in out._sumw:
                out._sumw[sparse_key] += self._sumw[sparse_key][dense_idx]
                if self._sumw2 is not None:
                    out._sumw2[sparse_key] += self._sumw2[sparse_key][dense_idx]
            else:
                out._sumw[sparse_key] = self._sumw[sparse_key][dense_idx].copy()
                if self._sumw2 is not None:
                    out._sumw2[sparse_key] = self._sumw2[sparse_key][dense_idx].copy()
        return out

    def fill(self, **values):
        if not all(d.name in values for d in self._axes):
            raise ValueError("Not all axes specified for this histogram!")

        if "weight" in values and self._sumw2 is None:
            self._init_sumw2()

        sparse_key = tuple(d.index(values[d.name]) for d in self.sparse_axes())
        if sparse_key not in self._sumw:
            self._sumw[sparse_key] = np.zeros(shape=self._dense_shape, dtype=self._dtype)
            if self._sumw2 is not None:
                self._sumw2[sparse_key] = np.zeros(shape=self._dense_shape, dtype=self._dtype)

        if self.dense_dim() > 0:
            dense_indices = tuple(d.index(values[d.name]) for d in self._axes if isinstance(d, Bin))
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

    def sum(self, axis, overflow='all'):
        """
            Projects current histogram down one dimension, summing along an axis
                axis: axis to integrate (either name or Axis object)
                overflow: 'none', 'all', 'nonan' (only applies to dense axes)
        """
        axis = self.axis(axis)
        reduced_dims = [ax for ax in self._axes if ax != axis]
        out = Hist(self._label, *reduced_dims, dtype=self._dtype)
        if self._sumw2 is not None:
            out._init_sumw2()
        if isinstance(axis, Bin):
            s = [slice(None)]*self.dense_dim()
            idense = self._idense(axis)
            if overflow == 'none':
                s[idense] = slice(1, -2)
            elif overflow == 'nonan':
                s[idense] = slice(1, -1)
            s = tuple(s)
            for key in self._sumw.keys():
                out._sumw[key] = np.sum(self._sumw[key][s], axis=idense)
                if self._sumw2 is not None:
                    out._sumw2[key] = np.sum(self._sumw2[key][s], axis=idense)
            return out
        elif isinstance(axis, Cat):
            isparse = self._isparse(axis)
            for key in self._sumw.keys():
                new_key = key[:isparse] + key[isparse+1:]
                if new_key in out._sumw:
                    out._sumw[new_key] += self._sumw[key]
                    if self._sumw2 is not None:
                        out._sumw2[new_key] += self._sumw2[key]
                else:
                    out._sumw[new_key] = self._sumw[key].copy()
                    if self._sumw2 is not None:
                        out._sumw2[new_key] = self._sumw2[key].copy()
            return out

    def project(self, axis_name, the_slice=slice(None)):
        """
            Projects current histogram down one dimension
                axis_name: dimension to reduce on
                the_slice: any slice, list, string, or other object that the axis will understand
        """
        axis = self.axis(axis_name)
        full_slice = tuple(slice(None) if ax != axis else the_slice for ax in self._axes)
        return self[full_slice].sum(axis)

    def profile(self, axis_name):
        raise NotImplementedError("Profiling along an axis")

    # TODO: multi-axis?
    def rebin_sparse(self, new_axis, old_axis, mapping):
        """
            Rebin sparse dimension(s)
                new_axis: A new sparse dimension
                old_axis: axis or name of axis which is being re-binned
                mapping: dictionary of {'new_bin': ['old_bin_1', 'old_bin_2', ...], ...}
        """
        if not isinstance(new_axis, Cat):
            raise TypeError("New axis must be a sparse axis")
        isparse = self._isparse(old_axis)
        new_dims = self._axes[:isparse] + (new_axis,) + self._axes[isparse+1:]
        out = Hist(self._label, *new_dims, dtype=self._dtype)
        if self._sumw2 is not None:
            out._init_sumw2()
        for new_cat in mapping.keys():
            new_idx = new_axis.index(new_cat)
            old_indices = mapping[new_cat]
            for key in self._sumw.keys():
                if key[isparse] not in old_indices:
                    continue
                new_key = key[:isparse] + (new_idx,) + key[isparse+1:]
                if new_key in out._sumw:
                    out._sumw[new_key] += self._sumw[key]
                    if self._sumw2 is not None:
                        out._sumw2[new_key] += self._sumw2[key]
                else:
                    out._sumw[new_key] = self._sumw[key].copy()
                    if self._sumw2 is not None:
                        out._sumw2[new_key] = self._sumw2[key].copy()
        return out

    # TODO: replace with __getitem__ with all the usual fancy indexing
    def values(self, sumw2=False, overflow_view=slice(1,-2)):
        """
            Returns dict of (sparse axis, ...): frequencies
            sumw2: if True, frequencies is a tuple (sum weights, sum sqaured weights)
            overflow_view: pass a slice object to control if underflow (0), overflow(-1), or nanflow(-2) are included
        """
        def view_dim(arr):
            if self.dense_dim() == 0:
                return arr
            else:
                return arr[tuple(overflow_view for _ in range(self.dense_dim()))]

        out = {}
        for sparse_key in self._sumw.keys():
            if sumw2:
                if self._sumw2 is not None:
                    w2 = view_dim(self._sumw2[sparse_key])
                else:
                    w2 = view_dim(self._sumw[sparse_key])
                out[sparse_key] = (view_dim(self._sumw[sparse_key]), w2)
            else:
                out[sparse_key] = view_dim(self._sumw[sparse_key])
        return out

    def scale(self, factor, axis=None):
        if self._sumw2 is None:
            self._init_sumw2()
        if isinstance(factor, numbers.Number) and axis is None:
            for key in self._sumw.keys():
                self._sumw[key] *= factor
                self._sumw2[key] *= factor**2
        elif isinstance(factor, dict):
            axis = self.axis(axis)
            isparse = self._isparse(axis)
            for key in self._sumw.keys():
                if key[isparse] in factor:
                    self._sumw[key] *= factor[key[isparse]]
                    self._sumw2[key] *= factor[key[isparse]]**2
        elif isinstance(factor, np.ndarray):
            axis = self.axis(axis)
            raise NotImplementedError("Scale dense dimension by a factor")
        else:
            raise TypeError("Could not interpret scale factor")
