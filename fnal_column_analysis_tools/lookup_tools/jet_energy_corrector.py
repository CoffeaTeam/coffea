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
                                                   
    def _evaluate(self,*args):        
        raise NotImplementedError
        
    def __repr__(self):
        out  = 'binned dims: %s\n'%(self._dim_order)
        out += 'eval vars  : %s\n'%(self._eval_vars)
        out += 'parameters : %s\n'%(self._parm_order)
        out += 'formula    : %s\n'%(self._formula_str)
        return out
