from ..lookup_tools.jme_standard_function import jme_standard_function
import warnings
import re
from ..util import awkward
from ..util import numpy as np
from copy import deepcopy


def _checkConsistency(against, tocheck):
    if against is None:
        against = tocheck
    else:
        if against != tocheck:
            raise Exception('Corrector for {} is mixed'
                            'with correctors for {}!'.format(tocheck, against))
    return tocheck


_levelre = re.compile('[L1-7]+')


def _getLevel(levelName):
    matches = _levelre.findall(levelName)
    if len(matches) > 1:
        raise Exception('Malformed JEC level name: {}'.format(levelName))
    return matches[0]


_level_order = ['L1', 'L2', 'L3', 'L2L3']


class FactorizedJetCorrector(object):
    """
    This class is a columnar implementation of the FactorizedJetCorrector tool in
    CMSSW and FWLite. It applies a series of JECs in ascending order as defined by
    '_level_order', and checks for the consistency of input corrections.

    It implements the jet energy correction definition specified in the JEC TWiki_.

    .. _TWiki: https://twiki.cern.ch/twiki/bin/view/CMS/JetEnergyScale

    You can use this class as follows::

        fjc = FactorizedJetCorrector(name1=corrL1,...)
        jetCorrs = fjc(JetParameter1=jet.parameter1,...)

    """
    def __init__(self, **kwargs):
        """
        You construct a FactorizedJetCorrector by passing in a dict of names and functions.
        Names must be formatted as '<campaign>_<dataera>_<datatype>_<level>_<jettype>'.
        """
        jettype = None
        levels = []
        funcs = []
        datatype = None
        campaign = None
        dataera = None
        for name, func in kwargs.items():
            if not isinstance(func, jme_standard_function):
                raise Exception('{} is a {} and not a jme_standard_function!'.format(name,
                                                                                     type(func)))
            info = name.split('_')
            if len(info) > 6 or len(info) < 5:
                raise Exception('Corrector name is not properly formatted!')
            offset = len(info) - 5

            campaign = _checkConsistency(campaign, info[0])
            dataera = _checkConsistency(dataera, info[1])
            datatype = _checkConsistency(datatype, info[2 + offset])
            levels.append(info[3 + offset])
            funcs.append(func)
            jettype = _checkConsistency(jettype, info[4 + offset])

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

        n_levels = len(self._levels)
        for i, level in enumerate(self._levels):
            this_level = _getLevel(level)
            ord_idx = _level_order.index(this_level)
            if i != this_level and n_levels > 1:
                self._levels[i], self._levels[ord_idx] = self._levels[ord_idx], self._levels[i]
                self._funcs[i], self._funcs[ord_idx] = self._funcs[ord_idx], self._funcs[i]

        # now we setup the call signature for this factorized JEC
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
        out = 'campaign   : %s\n' % (self._campaign)
        out += 'data era   : %s\n' % (self._dataera)
        out += 'data type  : %s\n' % (self._datatype)
        out += 'jet type   : %s\n' % (self._jettype)
        out += 'levels     : %s\n' % (','.join(self._levels))
        out += 'signature  : (%s)\n' % (','.join(self._signature))
        return out

    def getCorrection(self, **kwargs):
        """
        Returns the set of corrections for all input jets at the highest available level

        Use it like::

            jecs = corrector.getCorrection(JetProperty1=jet.property1,...)

        """
        subCorrs = self.getSubCorrections(**kwargs)
        return subCorrs[-1]

    def getSubCorrections(self, **kwargs):
        """
        Returns the set of corrections for all input jets broken down by level

        Use it like::

            jecs = corrector.getSubCorrections(JetProperty1=jet.property1,...)
            #'jecs' will be formatted like [[jec_jet1 jec_jet2 ...] ...]

        """
        localargs = {}
        localargs.update(kwargs)
        corrVars = []
        if 'JetPt' in kwargs.keys():
            corrVars.append('JetPt')
            localargs['JetPt'] = kwargs['JetPt'].copy()
        if 'JetE' in kwargs.keys():
            corrVars.append('JetE')
            localargs['JetE'] = kwargs['JetE'].copy()
        if len(corrVars) == 0:
            raise Exception('No variable to correct, need JetPt or JetE in inputs!')
        firstarg = localargs[self._signature[0]]
        cumulativeCorrection = 1.0
        counts = None
        if isinstance(firstarg, awkward.JaggedArray):
            counts = firstarg.counts
            cumulativeCorrection = firstarg.ones_like().flatten()
            for key in localargs.keys():
                localargs[key] = localargs[key].flatten().copy()
        else:
            cumulativeCorrection = np.ones_like(firstarg)
        corrections = []
        for i, func in enumerate(self._funcs):
            sig = func.signature
            args = []
            for input in sig:
                args.append(localargs[input])
            corr = func(*tuple(args))
            for var in corrVars:
                localargs[var] *= corr
            cumulativeCorrection *= corr
            corrections.append(cumulativeCorrection)
        if counts is not None:
            for i in range(len(corrections)):
                corrections[i] = awkward.JaggedArray.fromcounts(counts, corrections[i])
        return corrections
