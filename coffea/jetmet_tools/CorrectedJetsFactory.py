from .JECStack import JECStack
from coffea.util import awkward
from coffea.util import awkward1
from coffea.util import numpy as np
from collections import namedtuple
import warnings
from copy import copy
from functools import partial

_stack_parts = ['jec', 'junc', 'jer', 'jersf']
_MIN_JET_ENERGY = 1e-2


def rand_gauss_ak0(arr, var):
    item = arr[var]
    wrap, _ = awkward.util.unwrap_jagged(item,
                                         item.JaggedArray,
                                         (item, ))
    return wrap(np.random.normal(size=item.content.size))


def rand_gauss_ak1(arr, var):
    item = arr[var]

    def getfunction(layout, depth):
        if (isinstance(layout, awkward1.layout.NumpyArray)
            or not isinstance(layout, (awkward1.layout.Content,
                                       awkward1.partition.PartitionedArray)
           )):
            return lambda: awkward1.layout.NumpyArray(np.random.normal(size=awkward1.count(item)))
        return None

    out = awkward1._util.recursively_apply(
        awkward1.operations.convert.to_layout(item),
        getfunction)
    assert out is not None
    return awkward1._util.wrap(out, awkward1._util.behaviorof(item))


def jer_smear(arr, variation, forceStochastic, ptGenJet, ptJet, etaJet):
    pt_gen = arr[ptGenJet] if not forceStochastic else None
    jetPt = arr[ptJet]

    if isinstance(jetPt, awkward.array.base.AwkwardArray):
        if forceStochastic:
            wrap, arrays = awkward.util.unwrap_jagged(jetPt,
                                                      jetPt.JaggedArray,
                                                      (jetPt, ))
            pt_gen = wrap(np.zeros_like(arrays[0]))
    elif isinstance(jetPt, awkward1.highlevel.Array):
        def getfunction(layout, depth):
            if (isinstance(layout, awkward1.layout.NumpyArray)
                or not isinstance(layout, (awkward1.layout.Content,
                                           awkward1.partition.PartitionedArray)
               )):
                return lambda: awkward1.layout.NumpyArray(np.zeros_like(size=awkward1.count(jetPt)))
            return None
        if forceStochastic:
            pt_gen = awkward1._util.recursively_apply(
                awkward1.operations.convert.to_layout(jetPt),
                getfunction)
            pt_gen = awkward1._util.wrap(pt_gen, awkward1._util.behaviorof(jetPt))
    else:
        raise Exception('\'arr\' must be an awkward array of some kind!')

    jersmear = arr['jet_energy_resolution'] * arr['jet_resolution_rand_gauss']
    jersf = arr['jet_energy_resolution_scale_factor'][:, :, variation]
    doHybrid = pt_gen > 0.

    detSmear = 1. + (jersf - 1.) * (-pt_gen + jetPt) / jetPt  # because of #367
    stochSmear = 1. + np.sqrt(np.maximum(jersf**2 - 1., 0.)) * jersmear

    min_jet_pt = _MIN_JET_ENERGY / np.cosh(arr[etaJet])
    min_jet_pt_corr = min_jet_pt / jetPt

    smearfact = None
    if isinstance(arr, awkward.array.base.AwkwardArray):
        wrap, arrays = awkward.util.unwrap_jagged(jetPt,
                                                  jetPt.JaggedArray,
                                                  (jetPt, min_jet_pt_corr))
        smearfact = np.where(doHybrid.content, detSmear.content, stochSmear.content)
        smearfact = np.where((smearfact * arrays[0]) < arrays[1], arrays[1], smearfact)
        smearfact = wrap(smearfact)
    elif isinstance(arr, awkward1.highlevel.Array):
        smearfact = awkward1.where(awkward1.flatten(doHybrid),
                                   awkward1.flatten(detSmear),
                                   awkward1.flatten(stochSmear))
        smearfact = awkward1.where((smearfact * awkward1.flatten(jetPt)) < awkward1.flatten(min_jet_pt),
                                   awkward1.flatten(min_jet_pt_corr),
                                   smearfact)

        def getfunction(layout, depth):
            if (isinstance(layout, awkward1.layout.NumpyArray)
                or not isinstance(layout, (awkward1.layout.Content,
                                           awkward1.partition.PartitionedArray)
               )):
                return lambda: awkward1.layout.NumpyArray(smearfact)
            return None
        smearfact = awkward1._util.recursively_apply(
            awkward1.operations.convert.to_layout(jetPt),
            getfunction)
        smearfact = awkward1._util.wrap(smearfact, awkward1._util.behaviorof(jetPt))
    else:
        raise Exception('\'arr\' must be an awkward array of some kind!')

    return smearfact


class JetUncertainty:
    def __init__(self, jets, has_jer, forceStochastic, name_map, VirtualType, uncertainty, lazy_cache):
        self.jets = jets
        self.has_jer = has_jer
        self.forceStochastic = forceStochastic
        self.name_map
        self.VirtualType = VirtualType
        self.uncertainty = uncertainty
        self.lazy_cache = lazy_cache
        if uncertainty is None:  # JER uncertainty are [central, up, down]
            self.UP = 1
            self.DOWN = 2
        else:  # JES uncertainties are [up, down]
            self.UP = 0
            self.DOWN = 1

    def build_unc(self, up_down):
        out = None
        if isinstance(self.jets, awkward.array.base.AwkwardArray):
            out = self.jets.copy()
        elif isinstance(self.jets, awkward1.highlevel.Array):
            out = copy(self.jets)
        else:
            raise Exception('\'jets\' must be an awkward array of some kind!')

        if self.uncertainty is None:  # JER uncertainties
            def jer_smeared_corr(arr, variation, forceStochastic, ptGenJet, ptJet, etaJet):
                return jer_smear(arr, variation, forceStochastic, ptGenJet, ptJet, etaJet)

            out['jet_energy_resolution_correction'] = self.VirtualType(jer_smeared_corr,
                                                                       args=(out, up_down, self.forceStochastic,
                                                                             self.name_map['ptGenJet'],
                                                                             self.name_map['JetPt'],
                                                                             self.name_map['JetEta']),
                                                                       length=len(out),
                                                                       cache=self.lazy_cache)

            def jer_smeared_val(arr, varName):
                return arr['jet_energy_resolution_correction'] * arr['jet_energy_correction'] * arr[varName]

            out[self.name_map['JetPt']] = self.VirtualType(jer_smeared_corr,
                                                           args=(out, self.name_map['ptRaw']),
                                                           length=len(out),
                                                           cache=self.lazy_cache)
            out[self.name_map['JetMass']] = self.VirtualType(jer_smeared_corr,
                                                             args=(out, self.name_map['massRaw']),
                                                             length=len(out),
                                                             cache=self.lazy_cache)
        else:  # JES uncertainties
            out['jet_energy_uncertainty'] = self.uncertainty

            def junc_smeared_val(arr, up_down, has_jer, varName):
                base = arr['jet_energy_correction'] * arr[varName]
                if has_jer:
                    base = arr['jet_energy_resolution_correction'] * base
                return out['jet_energy_uncertainty'][up_down] * base

            out[self.name_map['JetPt']] = self.VirtualType(jer_smeared_corr,
                                                           args=(out, up_down, self.has_jer,
                                                                 self.name_map['ptRaw'],
                                                                 ),
                                                           length=len(out),
                                                           cache=self.lazy_cache)
            out[self.name_map['JetMass']] = self.VirtualType(jer_smeared_corr,
                                                             args=(out, up_down, self.has_jer,
                                                                   self.name_map['massRaw']
                                                                   ),
                                                             length=len(out),
                                                             cache=self.lazy_cache)

        return out

    def up(self):
        return self.build_unc(self.UP)

    def down(self):
        return self.build_unc(self.DOWN)


class JetUncertaintyHandler:
    def __init__(self, jets, has_jer, forceStochastic, name_map, VirtualType, uncertainties=None, lazy_cache=None):
        self.jets = jets
        self.has_jer = has_jer
        self.forceStochastic = forceStochastic
        self.name_map = name_map
        self.uncertainties = None
        self.lazy_cache = lazy_cache
        self.VirtualType = VirtualType
        if uncertainties is not None:
            self.uncertainties = {k: v for k, v in uncertainties}

    def keys(self):
        out = []
        if self.has_jer:
            out += ['JER']
        if self.uncertainties is not None:
            out += list(self.uncertainties.keys())
        return out

    def __getitem__(self, key):
        if ((key == 'JER' and not self.has_jer) or
            (self.uncertainties is not None and key not in self.uncertainties)):
            raise Exception(f'{key} is not an available uncertainty!')
        func = None
        if self.uncertainties is not None:
            func = self.uncertainties[key]

        return JetUncertainty(self.jets, self.has_jer, self.forceStochastic, self.name_map, self.VirtualType, func, self.lazy_cache)


class CorrectedJetsFactory(object):

    def __init__(self, name_map, jec_stack):
        # from PhysicsTools/PatUtils/interface/SmearedJetProducerT.h#L283
        self.forceStochastic = False

        if 'ptRaw' not in name_map or name_map['ptRaw'] is None:
            warnings.warn('There is no name mapping for ptRaw,'
                          ' CorrectedJets will assume that <object>.pt is raw pt!')
            name_map['ptRaw'] = name_map['JetPt'] + '_raw'
        self.treat_pt_as_raw = 'ptRaw' not in name_map

        if 'massRaw' not in name_map or name_map['massRaw'] is None:
            warnings.warn('There is no name mapping for massRaw,'
                          ' CorrectedJets will assume that <object>.mass is raw pt!')
            name_map['ptRaw'] = name_map['JetMass'] + '_raw'

        total_signature = set()
        for part in _stack_parts:
            attr = getattr(jec_stack, part)
            if part is not None:
                total_signature.update(attr.signature)

        missing = total_signature - set(name_map.keys())
        if len(missing) > 0:
            raise Exception(f'Missing mapping of {missing} in name_map!' +
                            ' Cannot evaluate jet corrections!' +
                            ' Please supply mappings for these variables!')

        if 'ptGenJet' not in name_map:
            warnings.warn('Input JaggedCandidateArray must have "ptGenJet" in order to apply hybrid JER smearing method. Stochastic smearing will be applied.')
            self.forceStochastic = True

        self.real_sig = [v for k, v in name_map.items()]
        self.name_map = name_map
        self.jec_stack = jec_stack

    def build(self, jets, lazy_cache=None):
        fields = None
        out = None
        VirtualType = None
        rand_gauss_func = None
        if isinstance(jets, awkward.array.base.AwkwardArray):
            fields = jets.columns
            if len(jets.columns) == 0:
                raise Exception('Detected awkward0: \'jets\' must have attributes specified in jets.columns!')
            out = jets.copy()
            VirtualType = awkward.VirtualArray
            rand_gauss_func = rand_gauss_ak0
        elif isinstance(jets, awkward1.highlevel.Array):
            fields = awkward1.keys(jets)
            if len(fields) == 0:
                raise Exception('Detected awkward1: \'jets\' must have attributes specified by keys!')
            out = copy(jets)
            VirtualType = awkward1.virtual
            rand_gauss_func = rand_gauss_ak1
        else:
            raise Exception('\'jets\' must be an awkward array of some kind!')

        if len(fields) == 0:
            raise Exception('Empty record, please pass a jet object with at least {self.real_sig} defined!')

        # take care of nominal JEC (no JER if available)
        out[self.name_map['JetPt'] + '_old'] = out[self.name_map['JetPt']]
        out[self.name_map['JetMass'] + '_old'] = out[self.name_map['JetMass']]
        if self.treat_pt_as_raw:
            out[self.name_map['ptRaw']] = out[self.name_map['JetPt']]
            out[self.name_map['massRaw']] = out[self.name_map['JetMass']]

        # here we render the args to pass to the jec so the switches should have been done
        jec_args = {k: out[self.name_map[k]] for k in self.jec_stack.jec.signature}
        out['jet_energy_correction'] = self.jec_stack.jec.getCorrection(**jec_args, lazy_cache=lazy_cache)

        # finally the lazy binding to the JEC
        def jec_var_corr(arr, varName):
            print('calling jec_var_corr')
            return arr['jet_energy_correction'] * arr[varName]

        out[self.name_map['JetPt']] = VirtualType(jec_var_corr, args=(out, self.name_map['ptRaw']),
                                                  length=len(out), cache=lazy_cache)
        out[self.name_map['JetMass']] = VirtualType(jec_var_corr, args=(out, self.name_map['massRaw']),
                                                    length=len(out), cache=lazy_cache)

        if self.jec_stack.jer is not None and self.jec_stack.jersf is not None:
            jerargs = {k: out[self.name_map[k]] for k in self.jec_stack.jer.signature}
            out['jet_energy_resolution'] = self.jec_stack.jer.getResolution(**jerargs, lazy_cache=lazy_cache)

            jersfargs = {k: out[self.name_map[k]] for k in self.jec_stack.jersf.signature}
            out['jet_energy_resolution_scale_factor'] = self.jec_stack.jersf.getScaleFactor(**jersfargs, lazy_cache=lazy_cache)

            out['jet_resolution_rand_gauss'] = VirtualType(rand_gauss_func, args=(out, self.name_map['JetPt']),
                                                           length=len(out), cache=lazy_cache)

            def jer_smeared_corr(arr, variation, forceStochastic, ptGenJet, ptJet, etaJet):
                print('calling jer_smeared_corr')
                return jer_smear(arr, variation, forceStochastic, ptGenJet, ptJet, etaJet)

            out['jet_energy_resolution_correction'] = VirtualType(jer_smeared_corr,
                                                                  args=(out, 0, self.forceStochastic,
                                                                        self.name_map['ptGenJet'],
                                                                        self.name_map['JetPt'],
                                                                        self.name_map['JetEta']),
                                                                  length=len(out),
                                                                  cache=lazy_cache)

            def jer_smeared_val(arr, varName):
                print('calling jer_smeared_val')
                return arr['jet_energy_resolution_correction'] * arr['jet_energy_correction'] * arr[varName]

            out[self.name_map['JetPt']] = VirtualType(jer_smeared_val, args=(out, self.name_map['ptRaw']),
                                                      length=len(out), cache=lazy_cache)
            out[self.name_map['JetMass']] = VirtualType(jer_smeared_val, args=(out, self.name_map['massRaw']),
                                                        length=len(out), cache=lazy_cache)

        return out

        def build_uncertainties(self, corrected_jets, lazy_cache=None):
            fields = None
            VirtualType = None
            if isinstance(corrected_jets, awkward.array.base.AwkwardArray):
                if not isinstance(jets, awkward.Table):
                    raise Exception('Detected awkward0: \'corrected_jets\' must be an awkward.Table!')
                fields = list(jets.columns.keys())
                VirtualType = awkward.VirtualArray
            elif isinstance(corrected_jets, awkward1.highlevel.Array):
                if not isinstance(awkward1.type(jets.layout), awkward1.types.RecordType):
                    raise Exception('Detected awkward1: \'corrected_jets\' must be a RecordType!')
                fields = awkward1.keys(jets)
                VirtualType = awkward1.virtual
            else:
                raise Exception('\'corrected_jets\' must be an awkward array of some kind!')

            if 'jet_energy_correction' not in fields:
                raise Exception('corrected_jets does not contain jet energy corrections! '
                                'Please run CorrectedJetsFactory.build() on them first!')

            has_jer = False
            if 'jet_energy_resolution_correction' in fields:
                has_jer = True

            juncs = None
            if self.jec_stack.juncs is not None:
                juncargs = {k: corrected_jets[self.name_map[k]] for k in self.jec_stack.junc.signature}
                juncs = self.jec_stack.juncs.getUncertainty(**juncargs)

            return JetUncertaintyHandler(corrected_jets,
                                         has_jer=has_jer,
                                         forceStochastic=self.forceStochastic,
                                         name_map=self.name_map,
                                         VirtualType=VirtualType,
                                         uncertainties=juncs,
                                         lazy_cache=lazy_cache)
