from .lookup_base import lookup_base

from ..util import awkward
from ..util import numpy
from ..util import numpy as np
from copy import deepcopy

from scipy.special import erf
from numpy import sqrt, log, log10, exp, abs
from numpy import maximum as max
from numpy import minimum as min
from numpy import power as pow


def wrap_formula(fstr, varlist):
    """
        Convert function string to python function
        Supports only simple math for now
    """
    val = fstr
    try:
        val = float(fstr)
        fstr = 'np.full_like(%s,%f)' % (varlist[0], val)
    except ValueError:
        val = fstr
    lstr = "lambda %s: %s" % (",".join(varlist), fstr)
    func = eval(lstr)
    return func


def masked_bin_eval(dim1_indices, dimN_bins, dimN_vals):
    dimN_indices = np.empty_like(dim1_indices)
    for i in np.unique(dim1_indices):
        idx = np.where(dim1_indices == i)
        dimN_indices[idx] = np.clip(np.searchsorted(dimN_bins[i],
                                                    dimN_vals[idx],
                                                    side='right') - 1,
                                    0, len(dimN_bins[i]) - 2)
    return dimN_indices


# idx_in is a tuple of indices in increasing jaggedness
# idx_out is a list of flat indices
def flatten_idxs(idx_in, jaggedarray):
    """
        This provides a faster way to convert between tuples of
        jagged indices and flat indices in a jagged array's contents
    """
    if len(idx_in) == 0:
        return np.array([], dtype=np.int)
    idx_out = jaggedarray.starts[idx_in[0]]
    if len(idx_in) == 1:
        pass
    elif len(idx_in) == 2:
        idx_out += idx_in[1]
    else:
        raise Exception('jme_standard_function only works for two binning dimensions!')

    good_idx = (idx_out < jaggedarray.content.size)
    if((~good_idx).any()):
        input_idxs = tuple([idx_out[~good_idx]] +
                           [idx_in[i][~good_idx] for i in range(len(idx_in))])
        raise Exception('Calculated invalid index {} for'
                        ' array with length {}'.format(np.vstack(input_idxs),
                                                       jaggedarray.content.size))

    return idx_out


class jme_standard_function(lookup_base):
    """
        This class defines a lookup table for jet energy corrections and resolutions.
        The JEC and JER values can be looked up with a call as follows:
        jerc_lut = jme_standard_function()
        jercs = jerc_lut(JetProperty1=jet.property1,...)
        "jercs" will be of the same shape as the input jet properties.
        The list of required jet properties are given in jersf_lut.signature
    """
    def __init__(self, formula, bins_and_orders, clamps_and_vars, parms_and_orders):
        """
            The constructor takes the output of the "convert_jec(jr)_txt_file"
            text file converter, which returns a formula, bins, and parameter values.
        """
        super(jme_standard_function, self).__init__()
        self._dim_order = bins_and_orders[1]
        self._bins = bins_and_orders[0]
        self._eval_vars = clamps_and_vars[2]
        self._eval_clamp_mins = clamps_and_vars[0]
        self._eval_clamp_maxs = clamps_and_vars[1]
        self._parm_order = parms_and_orders[1]
        self._parms = parms_and_orders[0]
        self._formula_str = formula
        self._formula = wrap_formula(formula, self._parm_order + self._eval_vars)

        for binname in self._dim_order[1:]:
            binsaslists = self._bins[binname].tolist()
            self._bins[binname] = [np.array(bins) for bins in binsaslists]

        # get the jit to compile if we've got more than one bin dim
        if len(self._dim_order) > 1:
            masked_bin_eval(np.array([0, 0]), self._bins[self._dim_order[1]], np.array([0.0, 0.0]))

        # compile the formula
        argsize = len(self._parm_order) + len(self._eval_vars)
        some_ones = tuple([50 * np.ones(argsize) for i in range(argsize)])
        _ = self._formula(*some_ones)

        self._signature = deepcopy(self._dim_order)
        for eval in self._eval_vars:
            if eval not in self._signature:
                self._signature.append(eval)
        self._dim_args = {self._dim_order[i]: i for i in range(len(self._dim_order))}
        self._eval_args = {}
        for i, argname in enumerate(self._eval_vars):
            self._eval_args[argname] = i + len(self._dim_order)
            if argname in self._dim_args.keys():
                self._eval_args[argname] = self._dim_args[argname]

    def _evaluate(self, *args):
        """ jec/jer = f(args) """
        bin_vals = {argname: args[self._dim_args[argname]] for argname in self._dim_order}
        eval_vals = {argname: args[self._eval_args[argname]] for argname in self._eval_vars}

        # lookup the bins that we care about
        dim1_name = self._dim_order[0]
        dim1_indices = np.clip(np.searchsorted(self._bins[dim1_name],
                                               bin_vals[dim1_name],
                                               side='right') - 1,
                               0, self._bins[dim1_name].size - 2)
        bin_indices = [dim1_indices]
        for binname in self._dim_order[1:]:
            bin_indices.append(masked_bin_eval(bin_indices[0],
                                               self._bins[binname],
                                               bin_vals[binname]))

        bin_tuple = tuple(bin_indices)

        # get clamp values and clip the inputs
        eval_values = []
        for eval_name in self._eval_vars:
            clamp_mins = None
            if self._eval_clamp_mins[eval_name].content.size == 1:
                clamp_mins = self._eval_clamp_mins[eval_name].content[0]
            else:
                idxs = flatten_idxs(bin_tuple, self._eval_clamp_mins[eval_name])
                clamp_mins = self._eval_clamp_mins[eval_name].content[idxs]
                if isinstance(clamp_mins, awkward.JaggedArray):
                    if clamp_mins.content.size == 1:
                        clamp_mins = clamp_mins.content[0]
                    else:
                        clamp_mins = clamp_mins.flatten()
            clamp_maxs = None
            if self._eval_clamp_maxs[eval_name].content.size == 1:
                clamp_maxs = self._eval_clamp_maxs[eval_name].content[0]
            else:
                idxs = flatten_idxs(bin_tuple, self._eval_clamp_maxs[eval_name])
                clamp_maxs = self._eval_clamp_maxs[eval_name].content[idxs]
                if isinstance(clamp_maxs, awkward.JaggedArray):
                    if clamp_maxs.content.size == 1:
                        clamp_maxs = clamp_maxs.content[0]
                    else:
                        clamp_maxs = clamp_maxs.flatten()
            eval_values.append(np.clip(eval_vals[eval_name], clamp_mins, clamp_maxs))

        # get parameter values
        parm_values = []
        if len(self._parms) > 0:
            idxs = flatten_idxs(bin_tuple, self._parms[0])
            parm_values = [parm.content[idxs] for parm in self._parms]

        return self._formula(*tuple(parm_values + eval_values))

    @property
    def signature(self):
        """ the list of all needed jet properties to be passed as kwargs to this lookup """
        return self._signature

    def __repr__(self):
        out = 'binned dims: %s\n' % (self._dim_order)
        out += 'eval vars  : %s\n' % (self._eval_vars)
        out += 'parameters : %s\n' % (self._parm_order)
        out += 'formula    : %s\n' % (self._formula_str)
        out += 'signature  : (%s)\n' % (','.join(self._signature))
        return out
