import numpy as np
from awkward.array.jagged import JaggedArray
import awkward as ak
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
        #print inputs
        counts = None
        #print
        for i in xrange(len(inputs)):
            #print i,type(inputs[i])
            if isinstance(inputs[i], JaggedArray):
                if counts is not None and set(counts) != set(inputs[i].counts) and len(counts) == len(inputs[i].counts):
                    if counts == -1:
                        raise Exception('do not mix JaggedArrays and numpy arrays when calling denselookup')
                    else:
                        raise Exception('counts not uniform between all input jagged arrays!')
                counts = inputs[i].counts
                inputs[i] = inputs[i].flatten()
                #print type(inputs[i])
            elif isinstance(inputs[i], np.ndarray):
                counts = -1
        retval = self.__evaluate(*tuple(inputs))
        for arg in args:
            if isinstance(arg, JaggedArray):
                retval = JaggedArray.fromcounts(arg.counts,retval)
                break
        #print retval
        #print
        return retval
                                               
    
    def __evaluate(self,*args):        
        indices = [] 
        for arg in args: 
            if type(arg) == JaggedArray: raise Exception('JaggedArray in inputs')
        if self.__dimension == 1:
            indices.append(np.maximum(np.minimum(np.searchsorted(self.__axes, args[0], side='right')-1,self.__values.shape[0]-1),0))
        else:
            for dim in xrange(self.__dimension):
                #print self.__axes[dim], self.__values.shape
                indices.append(np.maximum(np.minimum(np.searchsorted(self.__axes[dim], args[dim], side='right')-1,self.__values.shape[len(self.__axes)-dim-1]-1),0))
        indices.reverse()
        return self.__values[tuple(indices)]
    
    def __repr__(self):
        myrepr = "{} dimensional histogram with axes:\n".format(self.__dimension)
        temp = "" 
        if self.__dimension == 1:
            temp = "\t1: {}\n".format(self.__axes)
        else:
            temp = "\t1: {}\n".format(self.__axes[0])
        for idim in xrange(1,self.__dimension):
            temp += "\t{}: {}\n".format(idim+1,self.__axes[idim])        
        myrepr += temp
        return myrepr

class evaluator(object):
    def __init__(self,names,primitives):
        self.__functions = {}
        for key, idx in names.iteritems():
            self.__functions[key] = denselookup(*primitives[idx])
            
    def __dir__(self):
        return self.__functions.keys()
        
    def __getitem__(self, key):
        return self.__functions[key]
        