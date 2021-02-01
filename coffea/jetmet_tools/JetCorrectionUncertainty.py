from coffea.lookup_tools.jec_uncertainty_lookup import jec_uncertainty_lookup
import warnings
import re
import awkward
import numpy
from copy import deepcopy


def _checkConsistency(against, tocheck):
    if against is None:
        against = tocheck
    else:
        if against != tocheck:
            raise Exception('Corrector for {} is mixed'
                            'with correctors for {}!'.format(tocheck, against))
    return tocheck


def split_jec_name(name):
    info = name.split('_')

    # Check for the case of regrouped jes uncertainties
    if "Regrouped" in info[0]:
        info.pop(0)
        if "UncertaintySources" in info:
            subinfo = info[info.index("UncertaintySources"):]
            if len(subinfo) == 4:
                info[-2:] = [subinfo[-2] + "_" + subinfo[-1]]

    # Check for the case when the dataera name contains a _ like "17Nov2017_V6"
    if re.match(r"V[0-9]+", info[2]):
        dataera = info[1] + info[2]
        info.pop(2)
        info[1] = dataera

    if "UncertaintySources" in info:
        lvl = info.pop()
        info[3] = lvl

    if len(info) != 5:
        raise Exception('Corrector name {0} is not properly formatted!'.format(name))

    campaign = _checkConsistency(None, info[0])
    dataera = _checkConsistency(None, info[1])
    datatype = _checkConsistency(None, info[2])
    level = info[3].replace('Uncertainty', 'jes')  # use a generic 'jes' for normal uncertainty
    jettype = _checkConsistency(None, info[4])

    return campaign, dataera, datatype, level, jettype


class JetCorrectionUncertainty(object):
    """
    This class is a columnar implementation of the JetCorrectionUncertainty tool in
    CMSSW and FWLite. It calculates the jet energy scale uncertainty for a corrected jet
    in a given binning.

    It implements the jet energy correction definition specified in the JES Uncertainty TWiki_.

    .. _TWiki: https://twiki.cern.ch/twiki/bin/view/CMS/JECUncertaintySources

    You can use this class as follows::

        jcu = JetCorrectionUncertainty(name1=corrL1,...)
        jetUncs = jcu(JetParameter1=jet.parameter1,...)

    """
    def __init__(self, **kwargs):
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
        for name, func in kwargs.items():
            if not isinstance(func, jec_uncertainty_lookup):
                raise Exception('{} is a {} and not a jec_uncertainty_lookup!'.format(name,
                                                                                      type(func)))
            campaign, dataera, datatype, level, jettype = split_jec_name(name)

            levels.append(level)
            funcs.append(func)

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

    @property
    def levels(self):
        """ list the different sources of uncertainty """
        return self._levels

    def __repr__(self):
        out = 'campaign   : %s\n' % (self._campaign)
        out += 'data era   : %s\n' % (self._dataera)
        out += 'data type  : %s\n' % (self._datatype)
        out += 'jet type   : %s\n' % (self._jettype)
        out += 'levels     : %s\n' % (','.join(self._levels))
        out += 'signature  : (%s)\n' % (','.join(self._signature))
        return out

    def getUncertainty(self, **kwargs):
        """
        Returns the set of uncertainties for all input jets for all the levels (== sources)

        Use it like::

            juncs = uncertainty.getUncertainty(JetProperty1=jet.property1,...)
            #'juncs' will be formatted like [('SourceName', [[up_val down_val]_jet1 ... ]), ...]
            #in a zip iterator

        """
        cache = kwargs.pop('lazy_cache', None)
        uncs = []
        for i, func in enumerate(self._funcs):
            sig = func.signature
            args = tuple(kwargs[input] for input in sig)

            if isinstance(args[0], awkward.highlevel.Array):
                uncs.append(awkward.virtual(func, args=args, length=len(args[0]), cache=cache))
            elif isinstance(args[0], numpy.ndarray):
                uncs.append(func(*args))  # np is non-lazy
            else:
                raise Exception('Unknown array library for inputs.')

        return zip(self._levels, uncs)
