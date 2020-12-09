from coffea.jetmet_tools.JECStack import JECStack
import awkward
import awkward1
import numpy as np
import warnings
from copy import copy


class CorrectedMETFactory(object):

    def __init__(self, name_map):
        if 'xMETRaw' not in name_map or name_map['xMETRaw'] is None:
            warnings.warn('There is no name mapping for ptRaw,'
                          ' CorrectedJets will assume that <object>.x is raw pt!')
            name_map['xMETRaw'] = name_map['METx'] + '_raw'
        self.treat_pt_as_raw = 'ptRaw' not in name_map

        if 'yMETRaw' not in name_map or name_map['yMETRaw'] is None:
            warnings.warn('There is no name mapping for massRaw,'
                          ' CorrectedJets will assume that <object>.x is raw pt!')
            name_map['yMETRaw'] = name_map['METy'] + '_raw'

        self.name_map = name_map

    def build(self, MET, corrected_jets, lazy_cache):
        if lazy_cache is None:
            raise Exception('CorrectedMETFactory requires a awkward-array cache to function correctly.')
        if (isinstance(MET, awkward.array.base.AwkwardArray) or
            isinstance(corrected_jets, awkward.array.base.AwkwardArray)):
            raise Exception('Awkward0 not supported by CorrectedMETFactory, please use the JetTransformer or awkward1')
        elif (not isinstance(MET, awkward1.highlevel.Array) or
              not isinstance(corrected_jets, awkward1.highlevel.Array)):
            raise Exception('\'MET\' must be an awkward array of some kind!')

        out = copy(MET)

        form = out[self.name_map['METpt']].layout.form
        length = len(out)

        orig_jets = copy(corrected_jets)
        orig_jets[self.name_map['JetPt']] = orig_jets[self.name_map['ptRaw']]
        orig_jets[self.name_map['JetMass']] = orig_jets[self.name_map['massRaw']]

        out['x_orig'] = getattr(out, self.name_map['METx'])
        out['y_orig'] = getattr(out, self.name_map['METy'])

        out[self.name_map['METpt'] + '_orig'] = out[self.name_map['METpt']]
        out[self.name_map['METphi'] + '_orig'] = out[self.name_map['METphi']]

        def corrected_met_cartesian(met, rawJets, corrJets, dim):
            return met[f'{dim}_orig'] - np.sum(getattr(rawJets, dim) - getattr(corrJets, dim))

        def corrected_met_cartesian_unc(met, rawJets, corrJets, dimMET, dimJets):
            return getattr(met, dimMET) - np.sum(getattr(rawJets, dimJets) - getattr(corrJets, dimJets))

        out['corrected_met_x'] = awkward1.virtual(
            corrected_met_cartesian,
            args=(out, orig_jets, corrected_jets, self.name_map['JETx']),
            length=length, form=form, cache=lazy_cache
        )
        out['corrected_met_y'] = awkward1.virtual(
            corrected_met_cartesian,
            args=(out, orig_jets, corrected_jets, self.name_map['JETy']),
            length=length, form=form, cache=lazy_cache
        )

        out[self.name_map['METpt']] = awkward1.virtual(
            lambda met: np.hypot(met['corrected_met_x'], met['corrected_met_y']),
            args=(out, ),
            length=length,
            form=form,
            cache=lazy_cache
        )
        out[self.name_map['METphi']] = awkward1.virtual(
            lambda met: np.arctan2(met['corrected_met_y'], met['corrected_met_x']),
            args=(out, ),
            length=length,
            form=form,
            cache=lazy_cache
        )

        def make_unclustered_variant(themet, op, deltaX, deltaY):
            variant = copy(themet)
            variant['corrected_met_x'] = awkward1.virtual(
                lambda met: op(out['corrected_met_x'], out[f'{deltaX}']),
                args=(out, ),
                length=length,
                form=form,
                cache=lazy_cache
            )
            variant['corrected_met_y'] = awkward1.virtual(
                lambda met: op(out['corrected_met_y'], out[f'{deltaY}']),
                args=(out, ),
                length=length,
                form=form,
                cache=lazy_cache
            )
            variant[self.name_map['METpt']] = awkward1.virtual(
                lambda met: np.hypot(out['corrected_met_x'], out['corrected_met_y']),
                args=(variant, ),
                length=length,
                form=form,
                cache=lazy_cache
            )
            variant[self.name_map['METphi']] = awkward1.virtual(
                lambda met: np.arctan2(out['corrected_met_y'], out['corrected_met_x']),
                args=(variant, ),
                length=length,
                form=form,
                cache=lazy_cache
            )
            return variant

        unclus_up = make_unclustered_variant(MET, lambda x, y: x + y,
                                             self.name_map['UnClusteredEnergyDeltaX'],
                                             self.name_map['UnClusteredEnergyDeltaY'])
        unclus_down = make_unclustered_variant(MET, lambda x, y: x - y,
                                               self.name_map['UnClusteredEnergyDeltaX'],
                                               self.name_map['UnClusteredEnergyDeltaY'])
        out['MET_UnclusteredEnergy'] = awkward1.zip({'up': unclus_up, 'down': unclus_down},
                                                    depth_limit=1,
                                                    with_name='METSystematic')

        def make_variant(name, variation):
            variant = copy(MET)
            variant['corrected_met_x'] = awkward1.virtual(
                corrected_met_cartesian_unc,
                args=(out, orig_jets, variation, self.name_map['METx'], self.name_map['JETx']),
                length=length, form=form, cache=lazy_cache
            )
            variant['corrected_met_y'] = awkward1.virtual(
                corrected_met_cartesian_unc,
                args=(out, orig_jets, variation, self.name_map['METy'], self.name_map['JETy']),
                length=length, form=form, cache=lazy_cache
            )
            variant[self.name_map['METpt']] = awkward1.virtual(
                lambda met: np.hypot(met['corrected_met_x'], met['corrected_met_y']),
                args=(variant, ),
                length=length,
                form=form,
                cache=lazy_cache
            )
            variant[self.name_map['METphi']] = awkward1.virtual(
                lambda met: np.arctan2(met['corrected_met_y'], met['corrected_met_x']),
                args=(variant, ),
                length=length,
                form=form,
                cache=lazy_cache
            )
            return variant

        for unc in filter(lambda x: x.startswith(('JER', 'JES')), awkward1.fields(corrected_jets)):
            up = make_variant(unc, corrected_jets[unc].up)
            down = make_variant(unc, corrected_jets[unc].down)
            out[unc] = awkward1.zip({'up': up, 'down': down},
                                    depth_limit=1,
                                    with_name='METSystematic')
        return out

    def uncertainties(self):
        return ['MET_UnclusteredEnergy']
