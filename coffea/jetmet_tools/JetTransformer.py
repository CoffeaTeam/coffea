from .FactorizedJetCorrector import FactorizedJetCorrector
from .JetResolution import JetResolution
from .JetResolutionScaleFactor import JetResolutionScaleFactor
from .JetCorrectionUncertainty import JetCorrectionUncertainty

from ..analysis_objects.JaggedCandidateArray import JaggedCandidateArray

import numpy as np
from uproot_methods.classes.TLorentzVector import (
    TLorentzVectorArray,
    ArrayMethods,
    PtEtaPhiMassArrayMethods
)
from copy import deepcopy
from pdb import set_trace
import warnings

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

    It implements the recommendations from the CMS Jet/MET group JEC workbook_.

    .. _workbook: https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections?redirectedfrom=CMS.WorkBookJetEnergyCorrections

    .. note:: Only the stochastic smearing method is implemented at the moment.

    Parameters
    ----------
        jec:
            a `FactorizedJetCorrector` calculator object or ``None``
        junc:
            a `JetCorrectionUncertainty` calculator object or ``None``
        jer:
            a `JetResolution` calculator object or ``None``
        jersf:
            a `JetResolutionScaleFactor` calculator object or ``None``

    It uses the `FactorizedJetCorrector`, `JetResolution`, `JetResolutionScaleFactor`, and
    `JetCorrectionUncertainty` classes to calculate the ingredients for the final updated jet
    object, which will be modified **in place**.

    The jet object must be a "JaggedCandidateArray" and have the additional properties:
        - ptRaw
        - massRaw

    These will be used to reset the jet pT and mass, and then calculate the updated pTs and
    masses for various corrections and smearings.

    You can instantiate this class like::

        xformer = JetTransformer(jec=my_jec, junc=uncertianies, jer=resolution, jersf=smear)

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

        # from PhysicsTools/PatUtils/interface/SmearedJetProducerT.h#L283
        self.MIN_JET_ENERGY = 1e-2

    @property
    def uncertainties(self):
        return self._junc.levels if self._junc is not None else []

    def transform(self, jet, met=None, forceStochastic=False):
        """
        This is the main entry point for JetTransformer and acts on arrays of jet data in-place.

        **precondition** : jet is a JaggedCandidateArray with additional attributes

            - 'ptRaw'
            - 'massRaw'
            - 'ptGenJet' if using hybrid JER

        You can call transform like this::

            xformer.transform(jets)

        **postcondition** : jet.pt, jet.mass, jet.p4 are updated to represent the corrected jet based on the input correction set.
        """
        if not isinstance(jet, JaggedCandidateArray):
            raise Exception('Input data must be a JaggedCandidateArray!')
        if ('ptRaw' not in jet.columns or 'massRaw' not in jet.columns):
            raise Exception('Input JaggedCandidateArray must have "ptRaw" & "massRaw"!')
        if ('ptGenJet' not in jet.columns):
            warnings.warn('Input JaggedCandidateArray must have "ptGenJet" in order to apply hybrid JER smearing method. Stochastic smearing will be applied.')
            forceStochastic = True

        if met is not None:
            if 'p4' not in met.columns:
                raise Exception('Input met must have a p4 column!')
            if not (isinstance(met['p4'], ArrayMethods) or
                    isinstance(met['p4'], PtEtaPhiMassArrayMethods)):
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
            juncs = list((name, values) for name, values in juncs)

        # if there's a jer and sf to apply we have to update the momentum too
        # right now only use stochastic smearing
        if self._jer is not None and self._jersf is not None:
            args = {key: getattr(jet, _signature_map[key]).content for key in self._jer.signature}
            jer = self._jer.getResolution(**args)

            args = {key: getattr(jet, _signature_map[key]).content for key in self._jersf.signature}
            jersf = self._jersf.getScaleFactor(**args)

            jersmear = jer * np.random.normal(size=jer.size)

            ptGenJet = np.zeros_like(jet.pt.content) if forceStochastic else jet.ptGenJet.content

            doHybrid = ptGenJet > 0

            jsmear_cen = np.where(doHybrid,
                                  1 + (jersf[:, 0] - 1) * (jet.pt.content - ptGenJet) / jet.pt.content,
                                  1. + np.sqrt(np.maximum(jersf[:, 0]**2 - 1.0, 0)) * jersmear)

            jsmear_up = np.where(doHybrid,
                                 1 + (jersf[:, 1] - 1) * (jet.pt.content - ptGenJet) / jet.pt.content,
                                 1. + np.sqrt(np.maximum(jersf[:, 1]**2 - 1.0, 0)) * jersmear)

            jsmear_down = np.where(doHybrid,
                                   1 + (jersf[:, -1] - 1) * (jet.pt.content - ptGenJet) / jet.pt.content,
                                   1. + np.sqrt(np.maximum(jersf[:, -1]**2 - 1.0, 0)) * jersmear)

            # from PhysicsTools/PatUtils/interface/SmearedJetProducerT.h#L255-L264
            min_jet_pt = self.MIN_JET_ENERGY / np.cosh(jet.eta.content)
            min_jet_pt_corr = min_jet_pt / jet.pt.content
            jsmear_up = np.where(jsmear_up * jet.pt.content < min_jet_pt,
                                 min_jet_pt_corr,
                                 jsmear_up)
            jsmear_down = np.where(jsmear_down * jet.pt.content < min_jet_pt,
                                   min_jet_pt_corr,
                                   jsmear_down)
            jsmear_cen = np.where(jsmear_cen * jet.pt.content < min_jet_pt,
                                  min_jet_pt_corr,
                                  jsmear_cen)

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
        met['p4'].content._contents = TLorentzVectorArray.from_ptetaphim(
            np.sqrt(new_x**2 + new_y**2), 0,
            np.arctan2(new_y, new_x), 0
        ).content
        if 'MetUnclustEnUpDeltaX' in met.columns:
            px_up = met['p4'].x + met['MetUnclustEnUpDeltaX']
            py_up = met['p4'].y + met['MetUnclustEnUpDeltaY']
            met['pt_UnclustEn_up'] = np.sqrt(px_up**2 + py_up**2)
            met['phi_UnclustEn_up'] = np.arctan2(py_up, px_up)

            px_down = met['p4'].x - met['MetUnclustEnUpDeltaX']
            py_down = met['p4'].y - met['MetUnclustEnUpDeltaY']
            met['pt_UnclustEn_down'] = np.sqrt(px_down**2 + py_down**2)
            met['phi_UnclustEn_down'] = np.arctan2(py_down, px_down)

        if self._junc is not None:
            jets_sin = np.sin(jet['p4'].phi)
            jets_cos = np.cos(jet['p4'].phi)
            for name, _ in juncs:
                for shift in ['up', 'down']:
                    px = met['p4'].x - (initial_p4.x - jet['pt_{0}_{1}'.format(name, shift)] * jets_cos).sum()
                    py = met['p4'].y - (initial_p4.y - jet['pt_{0}_{1}'.format(name, shift)] * jets_sin).sum()
                    met['pt_{0}_{1}'.format(name, shift)] = np.sqrt(px**2 + py**2)
                    met['phi_{0}_{1}'.format(name, shift)] = np.arctan2(py, px)
