from ..lookup_tools.jec_uncertainty_lookup import jec_uncertainty_lookup
import warnings
import re
from fnal_column_analysis_tools.util import awkward
from fnal_column_analysis_tools.util import numpy as np
from copy import deepcopy

def _checkConsistency(against,tocheck):
    if against is None:
        against = tocheck
    else:
        if against != tocheck:
            raise Exception('Corrector for {} is mixed'/
                            'with correctors for {}!'.format(tocheck,against))
    return tocheck

_levelre = re.compile('Uncertainty')
def _getLevel(levelName):
    matches = _levelre.findall(levelName)
    if len(matches) != 1:
        raise Exception('Malformed JUNC level name: {}'.format(levelName))
    return matches[0]

_level_order = ['Uncertainty']

class JetCorrectionUncertainty(object):
    """
        This class is a columnar implementation of the JetCorrectionUncertainty tool in
        CMSSW and FWLite. It calculates the jet energy scale uncertainty for a corrected jet
        in a given binning.
        You can use this class as follows:
        jcu = JetCorrectionUncertainty(name1=corrL1,...)
        jetUncs = jcu(JetParameter1=jet.parameter1,...)
    """
    def __init__(self,**kwargs):
        """
            You construct a JetCorrectionUncertainty by passing in a dict of names and functions.
            Names must be formatted as '<campaign>_<dataera>_<datatype>_<level>_<jettype>'.
        """
        jettype = None
        levels = []
        funcs = []
        datatype = None
        campaign = None
        dataera = None
        for name,func in kwargs.items():
            if not isinstance(func,jec_uncertainty_lookup):
                raise Exception('{} is a {} and not a jec_uncertainty_lookup!'.format(name,
                                                                                      type(func)))
            info = name.split('_')
            if len(info) != 5:
                raise Exception('Corrector name is not properly formatted!')
            
            campaign = _checkConsistency(campaign,info[0])
            dataera  = _checkConsistency(dataera,info[1])
            datatype = _checkConsistency(datatype,info[2])
            levels.append(info[3])
            funcs.append(func)
            jettype  = _checkConsistency(jettype,info[4])
        
        if campaign is None:
            raise Exception('Unable to determine production campaign of JECs!')
        else:
            self._campaign = campaign
        
        if dataera is None:
            raise Exception('Unable to determine data era of JECs!')
        else:
            self._dataera = dataera
        
        if datatype is None:
            raise Exception('Unable to determine if JECs are for MC or Data!')
        else:
            self._datatype = datatype
        
        if len(levels) == 0:
            raise Exception('No levels provided?')
        else:
            self._levels = levels
            self._funcs = funcs
        
        if jettype is None:
            raise Exception('Unable to determine type of jet to correct!')
        else:
            self._jettype = jettype
        
        for i,level in enumerate(self._levels):
            this_level = _getLevel(level)
            ord_idx = _level_order.index(this_level)
            if i != this_level:
                self._levels[i],self._levels[ord_idx] = self._levels[ord_idx],self._levels[i]
                self._funcs[i],self._funcs[ord_idx] = self._funcs[ord_idx],self._funcs[i]
    
        #now we setup the call signature for this factorized JEC
        self._signature = []
        for func in self._funcs:
            sig = func.signature
            for input in sig:
                if input not in self._signature:
                    self._signature.append(input)

    @property
    def signature(self):
        """ list the necessary jet properties that must be input to this function """
        return self._signature
    
    def __repr__(self):
        out  = 'campaign   : %s\n'%(self._campaign)
        out += 'data era   : %s\n'%(self._dataera)
        out += 'data type  : %s\n'%(self._datatype)
        out += 'jet type   : %s\n'%(self._jettype)
        out += 'levels     : %s\n'%(','.join(self._levels))
        out += 'signature  : (%s)\n'%(','.join(self._signature))
        return out
    
    def getUncertainty(self,**kwargs):
        """
            Returns the set of uncertainties for all input jets at the highest available level
            use like:
            juncs = uncertainty.getUncertainty(JetProperty1=jet.property1,...)
            'juncs' will be formatted like [[[up_val down_val]_jet1 ... ] ...]
        """
        uncs = []
        for i,func in enumerate(self._funcs):
            sig = func.signature
            args = []
            for input in sig:
                args.append(kwargs[input])
            uncs.append(func(*tuple(args)))
        return uncs[-1]
