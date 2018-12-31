from fnal_column_analysis_tools.lookup_tools.lookup_base import lookup_base

import numpy as np
from awkward.array.jagged import JaggedArray
from copy import deepcopy
import numba

from numpy import sqrt,log,exp,abs
from numpy import maximum as max
from numpy import minimum as min
from numpy import power as pow

def numbaize(fstr, varlist):
    """
        Convert function string to numba function
        Supports only simple math for now
        """
    val = fstr
    try:
        val = float(fstr)
        fstr = 'np.full_like(%s,%f)' % ( varlist[0], val )
    except ValueError:
        val = fstr
    lstr = "lambda %s: %s" % (",".join(varlist), fstr)
    func = eval(lstr)
    nfunc = numba.njit(func)
    return nfunc

#@numba.njit ### somehow the unjitted form is ~20% faster?, and we can make fewer masks this way
def masked_bin_eval(dim1_indices, dimN_bins, dimN_vals):
    dimN_indices = np.empty_like(dim1_indices)
    for i in np.unique(dim1_indices):
        mask = (dim1_indices==i)
        dimN_indices[mask] = np.clip(np.searchsorted(dimN_bins[i],dimN_vals[mask],side='right')-1,
                                     0,len(dimN_bins[i])-1)
    return dimN_indices

class jme_standard_function(lookup_base):
    def __init__(self,formula,bins_and_orders,clamps_and_vars,parms_and_orders):
        super(jme_standard_function,self).__init__()
        self._dim_order = bins_and_orders[1]
        self._bins = bins_and_orders[0]
        self._eval_vars = clamps_and_vars[2]
        self._eval_clamp_mins = clamps_and_vars[0]
        self._eval_clamp_maxs = clamps_and_vars[1]
        self._parm_order = parms_and_orders[1]
        self._parms = parms_and_orders[0]
        self._formula_str = formula
        self._formula = numbaize(formula,self._parm_order+self._eval_vars)
    
        for binname in self._dim_order[1:]:
            binsaslists = self._bins[binname].tolist()
            self._bins[binname] = [np.array(bins) for bins in binsaslists]
        
        #get the jit to compile if we've got more than one bin dim
        if len(self._dim_order) > 1:
            masked_bin_eval(np.array([0,0]),self._bins[self._dim_order[1]],np.array([0.0,0.0]))
    
        #compile the formula
        argsize = len(self._parm_order) + len(self._eval_vars)
        some_ones = [50*np.ones(argsize) for i in range(argsize)]
        _ = self._formula(*tuple(some_ones))
    
        self._signature = deepcopy(self._dim_order)
        for eval in self._eval_vars:
            if eval not in self._signature:
                self._signature.append(eval)
        self._dim_args = {self._dim_order[i]:i for i in range(len(self._dim_order))}
        self._eval_args = {}
        for i,argname in enumerate(self._eval_vars):
            self._eval_args[argname] = i + len(self._dim_order)
            if argname in self._dim_args.keys():
                self._eval_args[argname] = self._dim_args[argname]

    def _evaluate(self,*args):
        bin_vals  = {argname:args[self._dim_args[argname]] for argname in self._dim_order}
        eval_vals = {argname:args[self._eval_args[argname]] for argname in self._eval_vars}
    
        #lookup the bins that we care about
        dim1_name = self._dim_order[0]
        dim1_indices = np.searchsorted(self._bins[dim1_name],bin_vals[dim1_name],side='right')
        dim1_indices = np.clip(dim1_indices-1,0,self._bins[dim1_name].size-1)
        bin_indices = [dim1_indices]
        for binname in self._dim_order[1:]:
            bin_indices.append(masked_bin_eval(bin_indices[0],
                                               self._bins[binname],
                                               bin_vals[binname]))
        bin_tuple = tuple(bin_indices)
        
        #get clamp values and clip the inputs
        eval_values = []
        for eval_name in self._eval_vars:
            clamp_mins = self._eval_clamp_mins[eval_name][bin_tuple]
            if isinstance(clamp_mins,JaggedArray):
                if clamp_mins.content.size == 1:
                    clamp_mins = clamp_mins.content[0]
                else:
                    clamp_mins = clamp_mins.flatten()
            clamp_maxs = self._eval_clamp_maxs[eval_name][bin_tuple]
            if isinstance(clamp_maxs,JaggedArray):
                if clamp_maxs.content.size == 1:
                    clamp_maxs = clamp_maxs.content[0]
                else:
                    clamp_maxs = clamp_maxs.flatten()
            eval_values.append(np.clip(eval_vals[eval_name],clamp_mins,clamp_maxs))

        #get parameter values
        parm_values = [parm[bin_tuple] for parm in self._parms]
        
        return self._formula(*tuple(parm_values+eval_values))
    
    @property
    def signature(self):
        return self._signature
    
    def __repr__(self):
        out  = 'binned dims: %s\n'%(self._dim_order)
        out += 'eval vars  : %s\n'%(self._eval_vars)
        out += 'parameters : %s\n'%(self._parm_order)
        out += 'formula    : %s\n'%(self._formula_str)
        out += 'signature  : (%s)\n'%(','.join(self._signature))
        return out
