from fnal_column_analysis_tools.lookup_tools.lookup_base import lookup_base

import numpy as np
from awkward.array.jagged import JaggedArray
from copy import deepcopy
import numba

from numpy import sqrt,log
from numpy import maximum as max

def numbaize(fstr, varlist):
    """
        Convert function string to numba function
        Supports only simple math for now
        """
    lstr = "lambda %s: %s" % (",".join(varlist), fstr)
    func = eval(lstr)
    nfunc = numba.njit(func)
    return nfunc

@numba.jit(nopython=False)
def masked_bin_eval(dim1_indices, dim1_size, dimN_bins, dimN_vals):
    dimN_indices = np.empty_like(dim1_indices)
    for i in range(dim1_size):
        dimN_indices[dim1_indices==i] = np.searchsorted(dimN_bins[i],dimN_vals[dim1_indices==i])
        dimN_indices[dim1_indices==i] = np.clip(dimN_indices[dimN_indices==i]-1,
                                                0,dimN_bins[i].size-1)
    return dimN_indices

class jet_energy_corrector(lookup_base):
    def __init__(self,formula,bins_and_orders,clamps_and_vars,parms_and_orders):
        super(jet_energy_corrector,self).__init__()
        self._dim_order = bins_and_orders[1]
        self._bins = bins_and_orders[0]
        self._eval_vars = clamps_and_vars[2]
        self._eval_clamp_mins = clamps_and_vars[0]
        self._eval_clamp_maxs = clamps_and_vars[1]
        self._parm_order = parms_and_orders[1]
        self._parms = parms_and_orders[0]
        self._formula_str = formula
        self._formula = numbaize(formula,self._parm_order+self._eval_vars)
    
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
        eval_vals = {argname:args[self._eval_args[argname]] for argname in self._eval_order}
    
        #lookup the bins that we care about
        dim1_name = bins_and_orders[0]
        dim1_indices = np.searchsorted(self._bins[dim1_name],args[bin_vals[dim1_name]],side='right')
        dim1_indices = np.clip(dim1_indices-1,0,self._bins[dim1_name].size-1)
        bin_indices = [dim1_indices]
        for binname in self._dim_order[1:]:
            bin_indices.append(masked_bin_eval(bin_indices[0],bin_indices[0].size,
                                               self._bins[binname],bin_vals[binname]))
        print(bin_indices)
        bin_tuple = tuple(bin_indices)
                
        #get clamp values and clip the inputs
        eval_values = []
        for eval_name in self._eval_order:
            clamp_mins = self._eval_clamp_mins[eval_name][bin_tuple]
            clamp_maxs = self._eval_clamp_maxs[eval_name][bin_tuple]
            eval_values.append(np.clip(eval_vals[eval_name],clamp_mins,clamp_maxs))

        #get parameter values
        parm_values = [parm[bin_tuple] for parm in self._parms]
        
        return self._formula(*tuple(parm_values+eval_values))
    
    def __repr__(self):
        out  = 'binned dims: %s\n'%(self._dim_order)
        out += 'eval vars  : %s\n'%(self._eval_vars)
        out += 'parameters : %s\n'%(self._parm_order)
        out += 'formula    : %s\n'%(self._formula_str)
        out += 'signature  : (%s)\n'%(','.join(self._signature))
        return out
