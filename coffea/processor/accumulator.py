from six import with_metaclass
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import numpy

try:
    from collections.abc import Set, Mapping
except ImportError:
    from collections import Set, Mapping


class AccumulatorABC(with_metaclass(ABCMeta)):
    '''ABC for an accumulator

    Derived class must implement:
        - ``identity()``: returns a new object of same type as self,
          such that ``self + self.identity() == self``
        - ``add(other)``: adds an object of same type as self to self

    Concrete implementations are then provided for ``__add__``, ``__radd__``, and ``__iadd__``
    '''
    @abstractmethod
    def identity(self):
        '''Identity of the accumulator

        A value such that any other value added to it will return
        the other value
        '''
        pass

    @abstractmethod
    def add(self, other):
        '''Add another accumulator to this one in-place'''
        pass

    def __add__(self, other):
        ret = self.identity()
        ret.add(self)
        ret.add(other)
        return ret

    def __radd__(self, other):
        ret = self.identity()
        ret.add(other)
        ret.add(self)
        return ret

    def __iadd__(self, other):
        self.add(other)
        return self


class value_accumulator(AccumulatorABC):
    '''Holds a value of arbitrary type

    Parameters
    ----------
        default_factory : callable
            a function that returns an instance of the desired identity value
        initial : bool, optional
            an initial value, if the identity is not the desired initial value
    '''
    def __init__(self, default_factory, initial=None):
        self.value = default_factory() if initial is None else initial
        self.default_factory = default_factory

    def __repr__(self):
        if type(self.default_factory) is type:
            defrepr = self.default_factory.__name__
        else:
            defrepr = repr(self.default_factory)
        return "value_accumulator(%s, %r)" % (defrepr, self.value)

    def identity(self):
        return value_accumulator(self.default_factory)

    def add(self, other):
        if isinstance(other, value_accumulator):
            self.value = self.value + other.value
        else:
            self.value = self.value + other


class list_accumulator(list, AccumulatorABC):
    '''A list with accumulator semantics

    See `list` for further info
    '''
    def identity(self):
        return list()

    def add(self, other):
        '''Add another accumulator to this one in-place'''
        if isinstance(other, list):
            list.extend(self, other)
        else:
            raise ValueError


class set_accumulator(set, AccumulatorABC):
    '''A set with accumulator semantics

    See `set` for further info
    '''
    def identity(self):
        return set_accumulator()

    def add(self, other):
        '''Add another accumulator to this one in-place

        Note
        ----
        This replaces `set.add` behavior, unfortunately.
        A workaround is to use `set.update`, e.g. ``a.update({'val'})``
        '''
        if isinstance(other, Set):
            set.update(self, other)
        else:
            set.add(self, other)


class dict_accumulator(dict, AccumulatorABC):
    '''A dictionary with accumulator semantics

    See `dict` for further info.
    It is assumed that the contents of the dict have accumulator semantics.
    '''
    def identity(self):
        ret = dict_accumulator()
        for key, value in self.items():
            ret[key] = value.identity()
        return ret

    def add(self, other):
        if isinstance(other, Mapping):
            for key, value in other.items():
                if key not in self:
                    if isinstance(value, AccumulatorABC):
                        self[key] = value.identity()
                    else:
                        raise ValueError
                self[key] += value
        else:
            raise ValueError


class defaultdict_accumulator(defaultdict, AccumulatorABC):
    '''A defaultdict with accumulator semantics

    See `collections.defaultdict` for further info.
    It is assumed that the contents of the dict have accumulator semantics
    '''
    def identity(self):
        return defaultdict_accumulator(self.default_factory)

    def add(self, other):
        for key, value in other.items():
            self[key] += value


class column_accumulator(AccumulatorABC):
    '''An appendable numpy ndarray

    Parameters
    ----------
        value : numpy.ndarray
            The identity value array, which should be an empty ndarray
            with the desired row shape. The column dimension will correspond to
            the first index of `value` shape.

    Examples
    --------
    If a set of accumulators is defined as::

        a = column_accumulator(np.array([]))
        b = column_accumulator(np.array([1., 2., 3.]))
        c = column_accumulator(np.array([4., 5., 6.]))

    then:

    >>> a + b
    column_accumulator(array([1., 2., 3.]))
    >>> c + b + a
    column_accumulator(array([4., 5., 6., 1., 2., 3.]))
    '''
    def __init__(self, value):
        if not isinstance(value, numpy.ndarray):
            raise ValueError("column_accumulator only works with numpy arrays")
        self._empty = numpy.zeros(dtype=value.dtype, shape=(0,) + value.shape[1:])
        self._value = value

    def __repr__(self):
        return "column_accumulator(%r)" % self.value

    def identity(self):
        return column_accumulator(self._empty)

    def add(self, other):
        if not isinstance(other, column_accumulator):
            raise ValueError("column_accumulator cannot be added to %r" % type(other))
        if other._empty.shape != self._empty.shape:
            raise ValueError("Cannot add two column_accumulator objects of dissimilar shape (%r vs %r)"
                             % (self._empty.shape, other._empty.shape))
        self._value = numpy.concatenate((self._value, other._value))

    @property
    def value(self):
        '''The current value of the column

        Returns a numpy array where the first dimension is the column dimension
        '''
        return self._value
