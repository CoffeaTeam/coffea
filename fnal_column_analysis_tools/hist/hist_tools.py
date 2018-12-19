from __future__ import division
import numbers
import math
import warnings
import copy
import re
import numpy as np


class Axis(object):
    # TODO: ABC? All derived must implement index(scalar), size(), equality
    def __init__(self, name, title):
        if name == "weight":
            raise ValueError("Cannot create axis: 'weight' is a reserved keyword for histograms")
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
            # Title doesn't matter
            return True
        elif isinstance(other, str):
            # Convenient for testing axis in list by name
            if self._name != other:
                return False
            return True
        raise TypeError("Cannot compare an Axis with a %r" % other)

    
class Cat(Axis):
    """
        Specify a category axis with name and title
            name: is used as a keyword in histogram filling, immutable
            title: describes the meaning of the axis, can be changed
        Number of categories is arbitrary, and filled sparsely
    """
    def __init__(self, name, title):
        super(Cat, self).__init__(name, title)
        self._categories = set()
    
    def index(self, scalar):
        if not isinstance(scalar, str):
            raise TypeError("Cat axis supports only string categories")
        # TODO: do we need some sort of hashing or just go by string?
        self._categories.add(scalar)
        return scalar
    
    @property
    def size(self):
        return len(self._categories)

    def __eq__(self, other):
        # Sparse, so as long as name is the same
        return super(Cat, self).__eq__(other)
        
    def _ireduce(self, pattern, regex=False):
        if not regex:
            pattern = "^" + pattern.replace('*', '.*')
        m = re.compile(pattern)
        return [v for v in self._categories if m.match(v)]


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
        super(Bin, self).__init__(name, title)
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
    def __init__(self, title, *axes, **kwargs):
        if not isinstance(title, str):
            raise TypeError("Title must be a string")
        self._title = title
        self._dtype = kwargs.pop('dtype', 'd')  # Much nicer in python3 :(
        if not all(isinstance(ax, Axis) for ax in axes):
            raise TypeError("All axes must be derived from Axis class")
        self._dense_dims = [ax for ax in axes if isinstance(ax, Bin)]
        self._dense_dims_shape = tuple([ax.size for ax in self._dense_dims])
        if np.prod(self._dense_dims_shape) > 1000000:
            warnings.warn("Allocating a large (>1M bin) histogram!", RuntimeWarning)
        # TODO: other sparse dimensions besides Cat could be used?
        self._sparse_dims = [ax for ax in axes if isinstance(ax, Cat)]
        self._sumw = {}
        # Storage of sumw2 starts at first use of weight keyword in fill()
        self._sumw2 = None
    
    def __repr__(self):
        return "<%s (%s) instance at 0x%0x>" % (self.__class__.__name__, ",".join(d.name for d in self.axes()), id(self))

    def copy(self):
        out = Hist(self._title, *(self._dense_dims + self._sparse_dims))
        out._sumw = copy.deepcopy(self._sumw)
        out._sumw2 = copy.deepcopy(self._sumw2)
        return out
        
    def clear(self):
        for key in self._sumw.keys():
            self._sumw[key].fill(0.)
        if self._sumw2 is not None:
            for key in self._sumw2.keys():
                self._sumw2[key].fill(0.)

    def _init_sumw2(self):
        self._sumw2 = {}
        for key in self._sumw.keys():
            self._sumw2[key] = self._sumw[key].copy()
    
    def fill(self, **values):
        if not all(d.name in values for d in self._dense_dims+self._sparse_dims):
            raise ValueError("Not all axes specified for this histogram!")
            
        if "weight" in values and self._sumw2 is None:
            self._init_sumw2()

        sparse_key = tuple(d.index(values[d.name]) for d in self._sparse_dims)
        if sparse_key not in self._sumw:
            self._sumw[sparse_key] = np.zeros(shape=self._dense_dims_shape, dtype=self._dtype)
            if self._sumw2 is not None:
                self._sumw2[sparse_key] = np.zeros(shape=self._dense_dims_shape, dtype=self._dtype)
            
        if self.dense_dim() > 0:
            dense_indices = tuple(d.index(values[d.name]) for d in self._dense_dims)
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
            for key in r.keys():
                if key in l:
                    l[key] += r[key]
                else:
                    l[key] = copy.deepcopy(r[key])
        
        add_dict(self._sumw, other._sumw)
        if self._sumw2 is not None and other._sumw2 is not None:
            add_dict(self._sumw2, other._sumw2)
        elif other._sumw2 is not None:
            self._sumw2 = copy.deepcopy(other._sumw2)
            for key in self._sumw:
                if key not in self._sumw2:
                    self._sumw2[key] = self._sumw[key].copy()
        return self
    
    def __add__(self, other):
        out = self.copy()
        out += other
        return out

    def sparse_dim(self):
        return len(self._sparse_dims)

    def dense_dim(self):
        return len(self._dense_dims)

    def project(self, axis_name, lo_hi=None, values=None, regex=None):
        """
            Projects current histogram down one dimension
                axis_name: dimension to reduce on
                lo_hi: if the dimension is dense, specify range as a tuple (inclusive of bounds)
                values: if the dimension is sparse, reduce according to either a:
                    value (wildcard * supported, or regular expression if regex=True)
                    list of values
        """
        if axis_name in self._dense_dims:
            if values is not None:
                raise ValueError("Use lo_hi= keyword argument for dense dimensions")
            iax = self._dense_dims.index(axis_name)
            reduced_dims = self._dense_dims[:iax] + self._dense_dims[iax+1:]
            out = Hist(self._title, *(self._sparse_dims + reduced_dims))
            if self._sumw2 is not None:
                out._init_sumw2()
            s = [slice(None) for _ in self._dense_dims]
            if lo_hi is not None:
                if not isinstance(lo_hi, tuple):
                    raise TypeError("Specify a tuple (lo, hi) when profiling dense dimensions (boundary is a closed interval)")
                indices = self._dense_dims[iax]._ireduce(lo_hi)
                s[iax] = slice(*indices)
            s = tuple(s)
            for key in self._sumw.keys():
                out._sumw[key] = np.sum(self._sumw[key][s], axis=iax)
                if self._sumw2 is not None:
                    out._sumw2[key] = np.sum(self._sumw2[key][s], axis=iax)
            return out
        elif axis_name in self._sparse_dims:
            if lo_hi is not None:
                raise ValueError("Use values= keyword argument for sparse dimensions")
            if values is None:
                values = "*"
            iax = self._sparse_dims.index(axis_name)
            reduced_dims = self._sparse_dims[:iax] + self._sparse_dims[iax+1:]
            out = Hist(self._title, *(reduced_dims + self._dense_dims))
            if self._sumw2 is not None:
                out._init_sumw2()
            if isinstance(values, str):
                values = self._sparse_dims[iax]._ireduce(values, regex)
            elif not isinstance(values, list):
                raise TypeError("Specify a values string or list of strings when profiling sparse dimensions")
            for key in self._sumw.keys():
                if key[iax] not in values:
                    continue
                new_key = key[:iax] + key[iax+1:]
                if new_key in out._sumw:
                    out._sumw[new_key] += self._sumw[key]
                    if self._sumw2 is not None:
                        out._sumw2[new_key] += self._sumw2[key]
                else:
                    out._sumw[new_key] = self._sumw[key].copy()
                    if self._sumw2 is not None:
                        out._sumw2[new_key] = self._sumw2[key].copy()
            return out
        raise ValueError("No axis named %s found in %r" % (axis_name, self))

    def profile(self, axis_name):
        raise NotImplementedError("Profiling along an axis")

    def axis(self, axis_name):
        if axis_name in self._dense_dims:
            return self._dense_dims[self._dense_dims.index(axis_name)]
        elif axis_name in self._sparse_dims:
            return self._sparse_dims[self._sparse_dims.index(axis_name)]
        raise ValueError("No axis named %s found in %r" % (axis_name, self))

    def axes(self):
        return self._sparse_dims + self._dense_dims

    # TODO: multi-axis
    def rebin_sparse(self, axis_name, new_name, new_title, mapping):
        iax = self._sparse_dims.index(axis_name)
        new_ax = Cat(new_name, new_title)
        new_dims = self._sparse_dims[:iax] + [new_ax,] + self._sparse_dims[iax+1:]
        out = Hist(self._title, *(new_dims + self._dense_dims))
        if self._sumw2 is not None:
            out._init_sumw2()
        for new_cat in mapping.keys():
            new_idx = new_ax.index(new_cat)
            old_indices = mapping[new_cat]
            for key in self._sumw.keys():
                if key[iax] not in old_indices:
                    continue
                new_key = key[:iax] + (new_idx,) + key[iax+1:]
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
        if isinstance(factor, numbers.Number) and axis is None:
            for key in self._sumw.keys():
                self._sumw[key] *= factor
                if self._sumw2:
                    self._sumw2[key] *= factor**2
        elif isinstance(factor, dict):
            if axis not in self._sparse_dims:
                raise ValueError("No named axis %s in %r" % (axis, self))
            iax = self._sparse_dims.index(axis)
            for key in self._sumw.keys():
                if key[iax] in factor:
                    self._sumw[key] *= factor[key[iax]]
                    if self._sumw2:
                        self._sumw2[key] *= factor[key[iax]]**2
        elif isinstance(factor, np.ndarray) and axis in self._dense_dims:
            raise NotImplementedError("Scale dense dimension by a factor")
        else:
            raise TypeError("Could not interpret scale factor")
