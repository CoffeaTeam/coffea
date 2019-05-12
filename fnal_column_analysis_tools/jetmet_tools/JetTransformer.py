from .FactorizedJetCorrector import FactorizedJetCorrector
from .JetResolution import JetResolution
from .JetResolutionScaleFactor import JetResolutionScaleFactor
from .JetCorrectionUncertainty import JetCorrectionUncertainty

from ..analysis_objects.JaggedCandidateArray import JaggedCandidateArray

import numpy as np
from uproot_methods import TLorentzVectorArray
from copy import deepcopy

_signature_map = {'JetPt': 'pt',
                  'JetEta': 'eta',
                  'Rho': 'rho',
                  'JetA': 'area'
                 }


def _update_jet_ptm(corr, jet, fromRaw=False):
    """
        This is a hack to update the jet pt and jet mass in place
        as we apply corrections and smearings.
    """
    if fromRaw:
        jet._content._contents['__fast_pt'] = corr * jet.ptRaw.content
        jet._content._contents['__fast_mass'] = corr * jet.massRaw.content
    else:
        jet._content._contents['__fast_pt'] = corr * jet.pt.content
        jet._content._contents['__fast_mass'] = corr * jet.mass.content


# the class below does use hacks of JaggedCandidateArray to achieve the desired behavior
# no monkey patches though
class JetTransformer(object):
    """
        This class is a columnar implementation of the the standard recipes for apply JECs, and
        the various scale factors and uncertainties therein.
           - Only the stochastic smearing method is implemented at the moment.
        It uses the FactorizedJetCorrector, JetResolution, JetResolutionScaleFactor, and
        JetCorrectionUncertainty classes to calculate the ingredients for the final updated jet
        object, which will be modified in place.
        The jet object must be a "JaggedCandidateArray" and have the additional properties:
           - ptRaw
           - massRaw
        These will be used to reset the jet pT and mass, and then calculate the updated pTs and
        masses for various corrections and smearings.
        You can use this class like:
        xformer = JetTransformer(name1=corrL1,...)
        xformer.transform(jet)
    """
    def __init__(self, jec=None, junc=None, jer=None, jersf=None):
        if jec is None:
            raise Exception('JetTransformer must have "jec" specified as an argument!')
        if not isinstance(jec, FactorizedJetCorrector):
            raise Exception('JetTransformer needs a FactorizedJetCorrecter passed as "jec"' +
                            ' got object of type {}'.format(type(jec)))
        self._jec = jec

        if junc is not None and not isinstance(junc, JetCorrectionUncertainty):
            raise Exception('"junc" must be of type "JetCorrectionUncertainty"' +
                            ' got {}'.format(type(junc)))
        self._junc = junc

        if (jer is None) != (jersf is None):
            raise Exception('Cannot apply JER-SF without an input JER, and vice-versa!')

        if jer is not None and not isinstance(jer, JetResolution):
            raise Exception('"jer" must be of type "JetResolution"' +
                            ' got {}'.format(type(jer)))
        self._jer = jer

        if jersf is not None and not isinstance(jersf, JetResolutionScaleFactor):
            raise Exception('"jersf" must be of type "JetResolutionScaleFactor"' +
                            ' got {}'.format(type(jersf)))
        self._jersf = jersf

    def transform(self, jet):
        """
            precondition - jet is a JaggedCandidateArray with additional attributes:
                             - 'ptRaw'
                             - 'massRaw'
            xformer = JetTransformer(name1=corrL1,...)
            xformer.transform(jet)
            postcondition - jet.pt, jet.mass, jet.p4 are updated to represent the corrected jet
                            based on the input correction set
        """
        if not isinstance(jet, JaggedCandidateArray):
            raise Exception('Input data must be a JaggedCandidateArray!')
        if ('ptRaw' not in jet.columns or 'massRaw' not in jet.columns):
            raise Exception('Input JaggedCandidateArray must have "ptRaw" & "massRaw"!')

        # initialize the jet momenta to raw values
        _update_jet_ptm(1.0, jet, fromRaw=True)

        # below we work in numpy arrays, JaggedCandidateArray knows how to convert them
        args = {key: getattr(jet, _signature_map[key]).content for key in self._jec.signature}
        jec = self._jec.getCorrection(**args)

        _update_jet_ptm(jec, jet, fromRaw=True)

        junc_up = np.ones_like(jec)
        junc_down = np.ones_like(jec)
        if self._junc is not None:
            args = {key: getattr(jet, _signature_map[key]).content for key in self._junc.signature}
            junc = self._junc.getUncertainty(**args)
            junc_up = junc[:, 0]
            junc_down = junc[:, 1]

        # if there's a jer and sf to apply we have to update the momentum too
        # right now only use stochastic smearing
        if self._jer is not None and self._jersf is not None:
            args = {key: getattr(jet, _signature_map[key]).content for key in self._jer.signature}
            jer = self._jer.getResolution(**args)

            args = {key: getattr(jet, _signature_map[key]).content for key in self._jersf.signature}
            jersf = self._jersf.getScaleFactor(**args)

            jersmear = jer * np.random.normal(size=jer.size)
            jsmear_cen = 1. + np.sqrt(jersf[:, 0]**2 - 1.0) * jersmear
            jsmear_up = 1. + np.sqrt(jersf[:, 1]**2 - 1.0) * jersmear
            jsmear_down = 1. + np.sqrt(jersf[:, -1]**2 - 1.0) * jersmear

            # need to apply up and down jer-smear before applying central correction
            jet.add_attributes(pt_jer_up=jsmear_up * jet.pt.content,
                               mass_jer_up=jsmear_up * jet.mass.content,
                               pt_jer_down=jsmear_down * jet.pt.content,
                               mass_jer_down=jsmear_down * jet.mass.content)
            # finally, update the central value
            _update_jet_ptm(jsmear_cen, jet)

        # have to apply central jersf before calculating junc
        if self._junc is not None:
            jet.add_attributes(pt_jes_up=junc_up * jet.pt.content,
                               mass_jes_up=junc_up * jet.mass.content,
                               pt_jes_down=junc_down * jet.pt.content,
                               mass_jes_down=junc_down * jet.mass.content)

        # hack to update the jet p4, we have the fully updated pt and mass here
        jet._content._contents['p4'] = TLorentzVectorArray.from_ptetaphim(jet.pt.content,
                                                                          jet.eta.content,
                                                                          jet.phi.content,
                                                                          jet.mass.content)
