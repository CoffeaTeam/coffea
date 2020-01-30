import numpy
import awkward
from uproot_methods.classes.TLorentzVector import PtEtaPhiMassArrayMethods
from uproot_methods.classes.TVector2 import ArrayMethods as XYArrayMethods


def _memoize(obj, name, constructor):
    memoname = '_memo_' + name
    if memoname not in obj.columns:
        out = constructor(obj)
        try:
            obj[memoname] = out
        except ValueError:
            # FIXME: add a column to view of the table
            return out
    return obj[memoname]


class METVector(XYArrayMethods):
    '''Implements the usual TVector2 methods, storing minimal materialized arrays in polar coordinates'''
    def __getitem__(self, key):
        if awkward.AwkwardArray._util_isstringslice(key) and key == 'fX':
            return _memoize(self, 'fX', lambda self: self['pt'] * numpy.cos(self['phi']))
        elif awkward.AwkwardArray._util_isstringslice(key) and key == 'fY':
            return _memoize(self, 'fY', lambda self: self['pt'] * numpy.sin(self['phi']))
        return super(METVector, self).__getitem__(key)

    # shortcut XYArrayMethods for pt and phi
    @property
    def pt(self):
        return self['pt']

    @property
    def phi(self):
        return self['phi']


class LorentzVector(PtEtaPhiMassArrayMethods):
    '''Implements the usual TLorentzVector methods, storing minimal materialized arrays in pt-eta-phi-mass coordinates'''
    _keymap = {'fPt': 'pt', 'fEta': 'eta', 'fPhi': 'phi', 'fMass': 'mass'}

    def __getitem__(self, key):
        if awkward.AwkwardArray._util_isstringslice(key) and key in self._keymap:
            if key == 'fMass' and 'mass' not in self.columns and 'fMass' not in self.columns:
                return _memoize(self, 'fMass', lambda self: self['pt'].zeros_like())
            elif 'fMass' not in self.columns:
                return self[self._keymap[key]]
        return super(LorentzVector, self).__getitem__(key)


class Candidate(LorentzVector):
    '''Implements the usual TLorentzVector methods but also propogates sum of charges'''
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = super(Candidate, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if ufunc is numpy.add and all(isinstance(i, Candidate) for i in inputs):
            out['charge'] = getattr(ufunc, method)(*(i['charge'] for i in inputs), **kwargs)
        # TODO else: type demotion?
        return out
