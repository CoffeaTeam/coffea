import numpy as np
from awkward.array.jagged import JaggedArray
from copy import deepcopy
import numba

### methods for dealing with b-tag SFs
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

### methods for dealing with b-tag SFs

class denselookup(object):
    def __init__(self,values,dims,feval_dim=None): 
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
                            'bytes' in values.dtype.name) #....
        if not isinstance(values, np.ndarray):
            raise TypeError("values is not a numpy array, but %r" % type(values))
        if vals_are_strings and feval_dim is None:
            raise Exception('Function objects passed to denselookup without knowing the arguments needed')
        elif vals_are_strings and feval_dim is not None: #convert strings to numba functions here
            funcs = np.zeros(shape=values.shape,dtype='O')
            for i in range(values.size):
                idx = np.unravel_index(i,dims=values.shape)
                funcs[idx] = numbaize(values[idx],['x'])
            self._values=deepcopy(funcs)
        else:
            self._values=deepcopy(values)
        # TODO: support for multidimensional functions and functions with variables other than 'x'
        if feval_dim is not None:
            if len(feval_dim) > 1:
                raise Exception('lookup_tools.evaluator only accepts 1D functions right now!')
            self._feval_dim = feval_dim[0]
    
    def __call__(self,*args):        
        inputs = list(args)
        offsets = None
        # TODO: check can use offsets (this should always be true for striped)
        # Alternatively we can just use starts and stops
        for i in range(len(inputs)):
            if isinstance(inputs[i], JaggedArray):
                if offsets is not None and offsets.base is not inputs[i].offsets.base:
                    if type(offsets) is int:
                        raise Exception('Do not mix JaggedArrays and numpy arrays when calling denselookup')
                    elif type(offsets) is np.ndarray and offsets.base is not inputs[i].offsets.base:
                        raise Exception('All input jagged arrays must have a common structure (offsets)!')
                offsets = inputs[i].offsets
                inputs[i] = inputs[i].content
            elif isinstance(inputs[i], np.ndarray):
                if offsets is not None:
                    if type(offsets) is np.ndarray:
                        raise Exception('do not mix JaggedArrays and numpy arrays when calling denselookup')
                offsets = -1
        retval = self._evaluate(*tuple(inputs))
        if offsets is not None and type(offsets) is not int:
            retval = JaggedArray.fromoffsets(offsets,retval)
        return retval
                                               
    def _evaluate(self,*args):        
        indices = [] 
        for arg in args: 
            if type(arg) == JaggedArray: raise Exception('JaggedArray in inputs')
        if self._dimension == 1:
            indices.append(np.clip(np.searchsorted(self._axes, args[0], side='right')-1,0,self._values.shape[0]-1))
        else:
            for dim in range(self._dimension):
                indices.append(np.clip(np.searchsorted(self._axes[dim], args[dim], side='right')-1,0,self._values.shape[len(self._axes)-dim-1]-1))
        indices.reverse()
        if self._feval_dim is not None:
            return numba_apply_1d(self._values[tuple(indices)], args[self._feval_dim])
        return self._values[tuple(indices)]
    
    def __repr__(self):
        myrepr = "{} dimensional histogram with axes:\n".format(self._dimension)
        temp = "" 
        if self._dimension == 1:
            temp = "\t1: {}\n".format(self._axes)
        else:
            temp = "\t1: {}\n".format(self._axes[0])
        for idim in range(1,self._dimension):
            temp += "\t{}: {}\n".format(idim+1,self._axes[idim])        
        myrepr += temp
        return myrepr

class evaluator(object):
    def __init__(self,names,primitives):
        self._functions = {}
        for key in names.keys():
            self._functions[key] = denselookup(*primitives[names[key]])
            
    def __dir__(self):
        return self._functions.keys()
        
    def __getitem__(self, key):
        return self._functions[key]
