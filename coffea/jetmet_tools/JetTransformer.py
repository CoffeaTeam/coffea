from .FactorizedJetCorrector import FactorizedJetCorrector
from .JetResolution import JetResolution
from .JetResolutionScaleFactor import JetResolutionScaleFactor
from .JetCorrectionUncertainty import JetCorrectionUncertainty

from ..analysis_objects.JaggedCandidateArray import JaggedCandidateArray

import numpy as np
from uproot_methods import TLorentzVectorArray
from copy import deepcopy
from pdb import set_trace

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

    @property
    def uncertainties(self):
        return self._junc.levels if self._junc is not None else []

    def transform(self, jet, met=None):
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

        if met is not None:
            if 'p4' not in met.columns:
                raise Exception('Input met must have a p4 column!')
            if not isinstance(met['p4'], TLorentzVectorArray):
                raise Exception('Met p4 must be a TLorentzVectorArray!')

        initial_p4 = jet['p4'].copy()  # keep a copy for fixing met
        # initialize the jet momenta to raw values
        _update_jet_ptm(1.0, jet, fromRaw=True)

        # below we work in numpy arrays, JaggedCandidateArray knows how to convert them
        args = {key: getattr(jet, _signature_map[key]).content for key in self._jec.signature}
        jec = self._jec.getCorrection(**args)

        _update_jet_ptm(jec, jet, fromRaw=True)

        juncs = None
        if self._junc is not None:
            args = {key: getattr(jet, _signature_map[key]).content for key in self._junc.signature}
            juncs = self._junc.getUncertainty(**args)

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
            for name, values in juncs:
                jet.add_attributes(**{
                    'pt_{0}_up'.format(name): values[:, 0] * jet.pt.content,
                    'mass_{0}_up'.format(name): values[:, 0] * jet.mass.content,
                    'pt_{0}_down'.format(name): values[:, 1] * jet.pt.content,
                    'mass_{0}_down'.format(name): values[:, 1] * jet.mass.content
                })

        # hack to update the jet p4, we have the fully updated pt and mass here
        jet._content._contents['p4'] = TLorentzVectorArray.from_ptetaphim(jet.pt.content,
                                                                          jet.eta.content,
                                                                          jet.phi.content,
                                                                          jet.mass.content)

        if met is None:
            return
        # set MET values
        new_x = met['p4'].x - (initial_p4.x - jet['p4'].x).sum()
        new_y = met['p4'].y - (initial_p4.y - jet['p4'].y).sum()
        met.base['p4'] = TLorentzVectorArray.from_ptetaphim(
            np.sqrt(new_x**2 + new_y**2), 0,
            np.arctan2(new_y, new_x), 0
        )
        if 'MetUnclustEnUpDeltaX' in met.columns:
            px_up = met['p4'].x + met['MetUnclustEnUpDeltaX']
            py_up = met['p4'].y + met['MetUnclustEnUpDeltaY']
            met.base['pt_UnclustEn_up'] = np.sqrt(px_up**2 + py_up**2)
            met.base['phi_UnclustEn_up'] = np.arctan2(py_up, px_up)

            px_down = met['p4'].x - met['MetUnclustEnUpDeltaX']
            py_down = met['p4'].y - met['MetUnclustEnUpDeltaY']
            met.base['pt_UnclustEn_down'] = np.sqrt(px_down**2 + py_down**2)
            met.base['phi_UnclustEn_down'] = np.arctan2(py_down, px_down)

        if self._junc is not None:
            jets_sin = np.sin(jet['p4'].phi)
            jets_cos = np.cos(jet['p4'].phi)
            for name, _ in juncs:
                for shift in ['up', 'down']:
                    px = met['p4'].x - (initial_p4.x - jet['pt_{0}_{1}'.format(name, shift)] * jets_cos).sum()
                    py = met['p4'].y - (initial_p4.y - jet['pt_{0}_{1}'.format(name, shift)] * jets_sin).sum()
                    met.base['pt_{0}_{1}'.format(name, shift)] = np.sqrt(px**2 + py**2)
                    met.base['phi_{0}_{1}'.format(name, shift)] = np.arctan2(py, px)
