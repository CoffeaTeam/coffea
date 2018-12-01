import numpy as np
from awkward.array.jagged import JaggedArray
from copy import deepcopy

class denselookup(object):
    def __init__(self,values,dims): 
        self.__dimension = 0
        whattype = type(dims)
        if whattype == np.ndarray:
            self.__dimension = 1
        else:
            self.__dimension = len(dims)        
        if self.__dimension == 0:
            raise Exception('Could not define dimension for {}'.format(whattype))
        self.__axes = deepcopy(dims)
        self.__values = deepcopy(values)
        self.__type = type(self.__values)
    
    def __call__(self,*args):        
        inputs = list(args)
        offsets = None
        # TODO: check can use offsets (this should always be true for striped)
        # Alternatively we can just use starts and stops
        for i in range(len(inputs)):
            if isinstance(inputs[i], JaggedArray):
                if offsets is not None and offsets.base is not inputs[i].offsets.base:
                    if type(offsets) is int:
                        raise Exception('do not mix JaggedArrays and numpy arrays when calling denselookup')
                    else:
                        raise Exception('All input jagged arrays must have a common structure (offsets)!')
                offsets = inputs[i].offsets
                inputs[i] = inputs[i].content
            elif isinstance(inputs[i], np.ndarray):
                offsets = -1
        retval = self.__evaluate(*tuple(inputs))
        if offsets is not None and type(offsets) is not int:
            retval = JaggedArray.fromoffsets(offsets,retval)
        return retval
                                               
    
    def __evaluate(self,*args):        
        indices = [] 
        for arg in args: 
            if type(arg) == JaggedArray: raise Exception('JaggedArray in inputs')
        if self.__dimension == 1:
            indices.append(np.clip(np.searchsorted(self.__axes, args[0], side='right')-1,0,self.__values.shape[0]-1))
        else:
            for dim in range(self.__dimension):
                #print self.__axes[dim], self.__values.shape
                indices.append(np.clip(np.searchsorted(self.__axes[dim], args[dim], side='right')-1,0,self.__values.shape[len(self.__axes)-dim-1]-1))
        indices.reverse()
        return self.__values[tuple(indices)]
    
    def __repr__(self):
        myrepr = "{} dimensional histogram with axes:\n".format(self.__dimension)
        temp = "" 
        if self.__dimension == 1:
            temp = "\t1: {}\n".format(self.__axes)
        else:
            temp = "\t1: {}\n".format(self.__axes[0])
        for idim in range(1,self.__dimension):
            temp += "\t{}: {}\n".format(idim+1,self.__axes[idim])        
        myrepr += temp
        return myrepr

class evaluator(object):
    def __init__(self,names,primitives):
        self.__functions = {}
        for key in names.keys():
            self.__functions[key] = denselookup(*primitives[names[key]])
            
    def __dir__(self):
        return self.__functions.keys()
        
    def __getitem__(self, key):
        return self.__functions[key]

