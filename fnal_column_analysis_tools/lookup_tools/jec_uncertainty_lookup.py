from fnal_column_analysis_tools.lookup_tools.lookup_base import lookup_base

import numpy as np
from awkward.array.jagged import JaggedArray
from copy import deepcopy
import numba

from scipy.interpolate import interp1d

from numpy import sqrt,log
from numpy import maximum as max
from numpy import minimum as min

@numba.njit
def masked_bin_eval(dim1_indices, dimN_bins, dimN_vals):
    dimN_indices = np.empty_like(dim1_indices)
    for i in np.unique(dim1_indices):
        dimN_indices[dim1_indices==i] = np.searchsorted(dimN_bins[i],dimN_vals[dim1_indices==i],side='right')
        dimN_indices[dim1_indices==i] = min(dimN_indices[dim1_indices==i]-1,len(dimN_bins[i])-1)
        dimN_indices[dim1_indices==i] = max(dimN_indices[dim1_indices==i]-1,0)
    return dimN_indices

class jec_uncertainty_lookup(lookup_base):
    def __init__(self,formula,bins_and_orders,knots_and_vars):
        super(jec_uncertainty_lookup,self).__init__()
        self._dim_order = bins_and_orders[1]
        self._bins = bins_and_orders[0]
        self._eval_vars = knots_and_vars[1]
        self._eval_knots = knots_and_vars[0]['knots']
        self._eval_downs = []
        self._eval_ups = []
        self._formula_str = formula.strip('"')
        self._formula = None
        if self._formula_str != 'None' and self._formula_str != '':
            raise Exception('jet energy uncertainties have no formula!')
    
        for binname in self._dim_order[1:]:
            binsaslists = self._bins[binname].tolist()
            self._bins[binname] = [np.array(bins) for bins in binsaslists]
        
        #convert downs and ups into interp1ds
        #(yes this only works for one binning dimension right now, fight me)
        for bin in range(self._bins[self._dim_order[0]].size-1):
            self._eval_downs.append(interp1d(self._eval_knots,
                                             knots_and_vars[0]['downs'][bin]))
            self._eval_ups.append(interp1d(self._eval_knots,
                                           knots_and_vars[0]['ups'][bin]))
        
        #get the jit to compile if we've got more than one bin dim
        if len(self._dim_order) > 1:
            masked_bin_eval(np.array([0]),self._bins[self._dim_order[1]],np.array([0.0]))
    
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
        print self._eval_args

    def _evaluate(self,*args):
        bin_vals  = {argname:args[self._dim_args[argname]] for argname in self._dim_order}
        eval_vals = {argname:args[self._eval_args[argname]] for argname in self._eval_vars}
    
        #lookup the bins that we care about
        dim1_name = self._dim_order[0]
        dim1_indices = np.searchsorted(self._bins[dim1_name],bin_vals[dim1_name],side='right')
        dim1_indices = np.clip(dim1_indices-1,0,self._bins[dim1_name].size-1)
        bin_indices = [dim1_indices]
        for binname in self._dim_order[1:]:
            bin_indices.append(masked_bin_eval(bin_indices[0],self._bins[binname],
                                               bin_vals[binname]))
        bin_tuple = tuple(bin_indices)
        
        #get clamp values and clip the inputs
        eval_ups = np.zeros_like(args[0])
        eval_downs = np.zeros_like(args[0])
        for i in np.unique(dim1_indices):
            eval_ups[dim1_indices==i] = self._eval_ups[i](eval_vals[self._eval_vars[0]][dim1_indices==i])
            eval_downs[dim1_indices==i] = self._eval_downs[i](eval_vals[self._eval_vars[0]][dim1_indices==i])
        
        central = np.ones_like(eval_ups)
        
        return np.vstack((central,1.+eval_ups,1.-eval_downs)).T
    
    def __repr__(self):
        out  = 'binned dims   : %s\n'%(self._dim_order)
        out += 'eval vars     : %s\n'%(self._eval_vars)
        out += 'signature     : (%s)\n'%(','.join(self._signature))
        return out
