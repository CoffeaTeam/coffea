from six import with_metaclass
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import numpy

try:
    from collections.abc import Set, Mapping
except ImportError:
    from collections import Set, Mapping


class AccumulatorABC(with_metaclass(ABCMeta)):
    '''
    ABC for an accumulator.  Derived must implement:
        identity: returns a new object of same type as self,
            such that self + self.identity() == self
        add(other): adds an object of same type as self to self

    Concrete implementations are provided for __add__, __radd__, __iadd__
    '''
    @abstractmethod
    def identity(self):
        pass

    @abstractmethod
    def add(self, other):
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
    '''
    Holds a value of arbitrary type, with identity
    as constructed by default_factory
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


class set_accumulator(set, AccumulatorABC):
    '''
    A set with accumulator semantics
    '''
    def identity(self):
        return set_accumulator()

    def add(self, other):
        if isinstance(other, Set):
            set.update(self, other)
        else:
            set.add(self, other)


class dict_accumulator(dict, AccumulatorABC):
    '''
    Like a dict but also has accumulator semantics
    It is assumed that the contents of the dict have accumulator semantics
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
    '''
    Like a defaultdict but also has accumulator semantics
    It is assumed that the contents of the dict have accumulator semantics
    '''
    def identity(self):
        return defaultdict_accumulator(self.default_factory)

    def add(self, other):
        for key, value in other.items():
            self[key] += value


class column_accumulator(AccumulatorABC):
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
        return self._value
