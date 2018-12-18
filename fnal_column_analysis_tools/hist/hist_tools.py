from __future__ import division
import numbers
import math
import warnings
import copy
import re
import numpy as np


class Axis(object):
    # TODO: ABC?
    # All derived must implement index(scalar), size(), equality
    def __init__(self, name, title):
        self._name = name
        self._title = title

    def __repr__(self):
        return "<%s (name=%s) instance at 0x%0x>" % (self.__class__.__name__, self._name, id(self))
    
    @property
    def name(self):
        return self._name
    
    @property
    def title(self):
        return self._title
    
    @title.setter
    def title(self, title):
        self._title = title

    def __eq__(self, other):
        if isinstance(other, Axis):
            if self._name != other._name:
                return False
            # TODO: Title?
            return True
        elif isinstance(other, str):
            # Convenience for finding axis name
            if self._name != other:
                return False
            return True
        raise ValueError("Cannot compare an Axis with a %r" % other)

    
class Cat(Axis):
    """
        Specify a category axis with name and title
            name: is used as a keyword in histogram filling, immutable
            title: describes the meaning of the axis, can be changed
        Number of categories is arbitrary, and filled sparsely
    """
    def __init__(self, name, title):
        super().__init__(name, title)
        self._categories = []
    
    def index(self, scalar):
        if not isinstance(scalar, str):
            raise ValueError("Cat axis supports only string categories")
        # TODO: SortedList?
        try:
            i = self._categories.index(scalar)
            return i
        except ValueError:
            i = len(self._categories)
            self._categories.append(scalar)
            return i
    
    def __getitem__(self, i):
        return self._categories[i]

    @property
    def size(self):
        return len(self._categories)

    def __eq__(self, other):
        # Sparse, so as long as name is the same
        return super().__eq__(other)
        
    def _ireduce(self, pattern, regex=False):
        if not regex:
            pattern = pattern.replace('*', '.*')
        m = re.compile(pattern)
        return [i for (i,v) in enumerate(self._categories) if m.match(v)]


class Bin(Axis):
    """
        Specify a binned axis with name and title, and binning
            name: is used as a keyword in histogram filling, immutable
            title: describes the meaning of the axis, can be changed
            n_or_arr: number of bins, if uniform binning, otherwise a list (or numpy 1D array) of bin boundaries
            lo: if uniform binning, minimum value
            hi: if uniform binning, maximum value
        Axis will generate frequencies for n+3 bins, special bin indices:
            0 = underflow, n+1 = overflow, n+2 = nanflow
        Bin boundaries are [lo, hi)
    """
    def __init__(self, name, title, n_or_arr, lo=None, hi=None):
        super().__init__(name, title)
        if isinstance(n_or_arr, (list, np.ndarray)):
            self._uniform = False
            self._bins = np.array(n_or_arr, dtype='d')
            if not all(np.sort(self._bins)==self._bins):
                raise ValueError("Binning not sorted!")
            
            self._lo = self._bins[0]
            self._hi = self._bins[-1]
            # to make searchsorted differentiate inf from nan
            np.append(self._bins, np.inf)
        elif isinstance(n_or_arr, (int, )) and lo and hi:
            self._uniform = True
            self._lo = lo
            self._hi = hi
            self._bins = n_or_arr
    
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
        raise ValueError("Request bin indices with a scalar or 1-D array only")
    
    @property
    def size(self):
        if self._uniform:
            return self._bins + 3
        # (inf added at constructor)
        return len(self._bins)+2

    def __eq__(self, other):
        if isinstance(other, Bin):
            if not super().__eq__(other):
                return False
            if self._uniform != other._uniform:
                return False
            if not ((self._uniform and self._bins==other._bins) or all(self._bins==other._bins)):
                return False
            return True
        return super().__eq__(other)

    def _ireduce(self, lo_hi):
        if self._uniform:
            blo = self.index(lo_hi[0])
            bhi = self.index(lo_hi[1])
            blo_ceil = np.clip(np.ceil((lo_hi[0]-self._lo)*self._bins/(self._hi-self._lo)) + 1, 0, self._bins+1)
            bhi_ceil = np.clip(np.ceil((lo_hi[1]-self._lo)*self._bins/(self._hi-self._lo)) + 1, 0, self._bins+1)
            if blo == 0 or bhi >= self.size-2:
                warnings.warn("Reducing along axis %r with a range [%f, %f] that exceeds bin boundaries" % ((self, )+lo_hi), RuntimeWarning)
            elif blo_ceil != blo or bhi_ceil != bhi:
                warnings.warn("Reducing along axis %r with a range [%f, %f] between bin boundaries, no interpolation is performed" % ((self,)+lo_hi), RuntimeWarning)
        else:
            if not (lo_hi[0] in self._bins and lo_hi[1] in self._bins):
                warnings.warn("Reducing along axis %r with a range [%f, %f] between bin boundaries, no interpolation is performed" % ((self,)+lo_hi), RuntimeWarning)
            blo = self.index(lo_hi[0])
            bhi = self.index(lo_hi[1]) - 1
        return (blo, bhi+1)

    def bin_boundaries(self, extended=False):
        if self._uniform:
            out = np.linspace(self._lo, self._hi, self._bins+1)
        else:
            out = self._bins.copy()
        if extended:
            out = np.insert(out, 0, -np.inf)
            out = np.append(out, np.inf)
            out = np.append(out, np.nan)
        return out

            
class Hist(object):
    """
        Specify a multidimensional histogram
            title: description of meaning of frequencies (axis descriptions specified in axis constructor)
            dtype: underlying numpy dtype of frequencies
            *axes: positional list of Cat or Bin objects
    """
    def __init__(self, title, *axes, dtype='d'):
        self._title = title
        self._dtype = dtype
        if not all(isinstance(ax, Axis) for ax in axes):
            raise TypeError("All axes must be derived from Axis class")
        self._dense_dims = [ax for ax in axes if isinstance(ax, Bin)]
        # TODO: handle no dense dims
        self._dense_dims_shape = tuple([ax.size for ax in self._dense_dims])
        if np.prod(self._dense_dims_shape) > 1000000:
            warnings.warn("Allocating a large (>1M bin) histogram!", RuntimeWarning)
        # TODO: other sparse dimensions besides Cat could be used
        self._sparse_dims = [ax for ax in axes if isinstance(ax, Cat)]
        self._sumw = {}
        self._sumw2 = {}
    
    def __repr__(self):
        return "<%s (%s) instance at 0x%0x>" % (self.__class__.__name__, ",".join(d.name for d in self._sparse_dims+self._dense_dims), id(self))

    def copy(self):
        out = Hist(self._title, *(self._dense_dims + self._sparse_dims))
        out._sumw = copy.deepcopy(self._sumw)
        out._sumw2 = copy.deepcopy(self._sumw2)
        return out
        
    def clear(self):
        for k in self._sumw.keys():
            self._sumw[k].fill(0.)
        for k in self._sumw2.keys():
            self._sumw2[k].fill(0.)
    
    def fill(self, **values):
        if not all(d.name in values for d in self._dense_dims+self._sparse_dims):
            raise ValueError("Not all axes specified for this histogram!")
            
        sparse_key = tuple(d.index(values[d.name]) for d in self._sparse_dims)
        if sparse_key not in self._sumw:
            self._sumw[sparse_key] = np.zeros(shape=self._dense_dims_shape, dtype=self._dtype)
        if "weight" in values and sparse_key not in self._sumw2:
            if sparse_key in self._sumw:
                self._sumw2[sparse_key] = self._sumw[sparse_key].copy()
            else:
                self._sumw2[sparse_key] = np.zeros(shape=self._dense_dims_shape, dtype=self._dtype)
            
        dense_indices = tuple(d.index(values[d.name]) for d in self._dense_dims)
        if "weight" in values:
            np.add.at(self._sumw[sparse_key], dense_indices, values["weight"])
            np.add.at(self._sumw2[sparse_key], dense_indices, values["weight"]**2)
        else:
            np.add.at(self._sumw[sparse_key], dense_indices, 1.)
    
    def __iadd__(self, other):
        if len(self._dense_dims)!=len(other._dense_dims):
            raise ValueError("Cannot add this histogram with histogram %r of dissimilar dense dimension" % other)
        if not all(d1==d2 for (d1,d2) in zip(self._dense_dims, other._dense_dims)):
            if set(d.name for d in self._dense_dims) == set(d.name for d in other._dense_dims):
                # Need to reorder dense array
                raise NotImplementedError("Adding histograms with shuffled dense dimensions")
            raise ValueError("Cannot add this histogram with histogram %r (mismatch in dense dimensions)" % other)
        if not all(d1==d2 for (d1,d2) in zip(self._sparse_dims, other._sparse_dims)):
            if set(d.name for d in self._sparse_dims) == set(d.name for d in other._sparse_dims):
                # Need to permute key order
                raise NotImplementedError("Adding histograms with shuffled sparse dimensions")
            raise ValueError("Cannot add this histogram with histogram %r (mismatch in sparse dimensions)" % other)

        def add_dict(l, r):
            for k in r.keys():
                if k in l:
                    l[k] += r[k]
                else:
                    l[k] = copy.deepcopy(r[k])
        
        # Prepare any missing sumw2
        for k in set(other._sumw2.keys()) - set(self._sumw2.keys()):
            if k in self._sumw:
                self._sumw2[k] = self._sumw[k]
        for k in set(self._sumw2.keys()) - set(other._sumw2.keys()):
            if k in other._sumw:
                self._sumw2[k] += other._sumw[k]

        add_dict(self._sumw, other._sumw)
        add_dict(self._sumw2, other._sumw2)
        return self
    
    def __add__(self, other):
        out = self.copy()
        out += other
        return out

    # TODO: could be sped up for multi-axis reduction
    def project(self, axis_name, lo_hi=None, pattern=None, regex=None):
        """
            Projects current histogram down one dimension
                axis_name: dimension to reduce on
                lo_hi: if the dimension is a dense dimension, specify range as a tuple (inclusive of bounds)
                pattern: if the dimension is sparse, reduce according to the pattern (wildcard * supported, or if regex=True, then any regular expression)
        """
        if axis_name in self._dense_dims:
            iax = self._dense_dims.index(axis_name)
            ax = self._dense_dims[iax]
            reduced_dims = self._dense_dims[:iax] + self._dense_dims[iax+1:]
            out = Hist(self._title, *(self._sparse_dims + reduced_dims))
            s = [slice(None) for _ in self._dense_dims]
            if lo_hi is not None:
                if not isinstance(lo_hi, tuple):
                    raise ValueError("Specify a tuple (lo, hi) when profiling dense dimensions (boundary is a closed interval)")
                indices = ax._ireduce(lo_hi)
                s[iax] = slice(*indices)
            s = tuple(s)
            for k in self._sumw.keys():
                out._sumw[k] = np.sum(self._sumw[k][s], axis=iax)
            for k in self._sumw2.keys():
                out._sumw2[k] = np.sum(self._sumw2[k][s], axis=iax)
            return out
        elif axis_name in self._sparse_dims:
            if pattern is None:
                pattern = "*"
            elif isinstance(pattern, str):
                iax = self._sparse_dims.index(axis_name)
                ax = self._sparse_dims[iax]
                indices = ax._ireduce(pattern, regex)
                reduced_dims = self._sparse_dims[:iax] + self._sparse_dims[iax+1:]
                out = Hist(self._title, *(self._dense_dims + reduced_dims))
                for k in self._sumw.keys():
                    new_key = k[:iax] + k[iax+1:]
                    if new_key in out._sumw:
                        out._sumw[new_key] += self._sumw[k]
                    else:
                        out._sumw[new_key] = copy.deepcopy(self._sumw[k])
                for k in self._sumw2.keys():
                    new_key = k[:iax] + k[iax+1:]
                    if new_key in out._sumw2:
                        out._sumw2[new_key] += self._sumw2[k]
                    else:
                        out._sumw2[new_key] = copy.deepcopy(self._sumw2[k])
                return out
            raise ValueError("Specify a search pattern string or list of strings when profiling sparse dimensions")
        raise ValueError("No axis named %s found in %r" % (axis_name, self))

    def profile(self, axis_name):
        raise NotImplementedError("Profiling along an axis")

    def axis(self, axis_name):
        if axis_name in self._dense_dims:
            return self._dense_dims[self._dense_dims.index[axis_name]]
        elif axis_name in self._sparse_dims:
            return self._sparse_dims[self._sparse_dims.index[axis_name]]
        raise ValueError("No axis named %s found in %r" % (axis_name, self))

    def values(self, sparse=True):
        if len(self._dense_dims) == 1:
            if sparse:
                out = {}
                for sparse_key in self._sumw.keys():
                    sparse_name = []
                    for d,i in zip(self._sparse_dims, sparse_key):
                        sparse_name.append(d[i])
                    # Chop overflow
                    out[tuple(sparse_name)] = self._sumw[sparse_key][1:-2]
                return out
            else:
                raise NotImplementedError("Make rectangular table for missing sparse dimensions")
        elif len(self._dense_dims) == 2:
            raise NotImplementedError("2D values formatted for plotting with matplotlib")
        else:
            raise NotImplementedError("Higher-than-two-dimensional values for plotting?")
