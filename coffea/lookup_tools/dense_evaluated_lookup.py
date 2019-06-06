from ..lookup_tools.lookup_base import lookup_base

from ..util import numpy as np
from ..util import awkward
from ..util import numba

from copy import deepcopy


# methods for dealing with b-tag SFs
@numba.jit(forceobj=True)
def numba_apply_1d(functions, variables):
    out = np.empty(variables.shape)
    for i in range(functions.size):
        out[i] = functions[i](variables[i])
    return out


def numbaize(fstr, varlist):
    """
        Convert function string to numba function
        Supports only simple math for now
        """
    lstr = "lambda %s: %s" % (",".join(varlist), fstr)
    func = eval(lstr)
    nfunc = numba.njit(func)
    return nfunc


# methods for dealing with b-tag SFs
class dense_evaluated_lookup(lookup_base):
    def __init__(self, values, dims, feval_dim=None):
        super(dense_evaluated_lookup, self).__init__()
        self._dimension = 0
        whattype = type(dims)
        if whattype == np.ndarray:
            self._dimension = 1
        else:
            self._dimension = len(dims)
        if self._dimension == 0:
            raise Exception('Could not define dimension for {}'.format(whattype))
        self._axes = deepcopy(dims)
        self._feval_dim = None
        vals_are_strings = ('string' in values.dtype.name or
                            'str' in values.dtype.name or
                            'unicode' in values.dtype.name or
                            'bytes' in values.dtype.name)  # ....
        if not isinstance(values, np.ndarray):
            raise TypeError('values is not a numpy array, but %r' % type(values))
        if not vals_are_strings:
            raise Exception('Non-string values passed to dense_evaluated_lookup!')
        if feval_dim is None:
            raise Exception('Evaluation dimensions not specified in dense_evaluated_lookup')
        funcs = np.zeros(shape=values.shape, dtype='O')
        for i in range(values.size):
            idx = np.unravel_index(i, shape=values.shape)
            funcs[idx] = numbaize(values[idx], ['x'])
        self._values = deepcopy(funcs)
        # TODO: support for multidimensional functions and functions with variables other than 'x'
        if len(feval_dim) > 1:
            raise Exception('lookup_tools.evaluator only accepts 1D functions right now!')
        self._feval_dim = feval_dim[0]

    def _evaluate(self, *args):
        indices = []
        for arg in args:
            if type(arg) == awkward.JaggedArray:
                raise Exception('JaggedArray in inputs')
        if self._dimension == 1:
            indices.append(np.clip(np.searchsorted(self._axes, args[0], side='right') - 1, 0, self._values.shape[0] - 1))
        else:
            for dim in range(self._dimension):
                indices.append(np.clip(np.searchsorted(self._axes[dim], args[dim], side='right') - 1,
                                       0, self._values.shape[len(self._axes) - dim - 1] - 1))
        indices.reverse()
        return numba_apply_1d(self._values[tuple(indices)], args[self._feval_dim])

    def __repr__(self):
        myrepr = "{} dimensional histogram with axes:\n".format(self._dimension)
        temp = ""
        if self._dimension == 1:
            temp = "\t1: {}\n".format(self._axes)
        else:
            temp = "\t1: {}\n".format(self._axes[0])
        for idim in range(1, self._dimension):
            temp += "\t{}: {}\n".format(idim + 1, self._axes[idim])
        myrepr += temp
        return myrepr
