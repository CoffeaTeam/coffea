from .common import LorentzVector
import awkward


class FatJet(LorentzVector):
    '''NanoAOD large radius jet object'''
    subjetmap = {'FatJet': 'SubJet'}  # V6 has 'GenJetAK8': 'SubGenJetAK8', maybe better to put in generator.py
    LOOSE = 0
    'jetId bit position'
    TIGHT = 1
    'jetId bit position'
    TIGHTLEPVETO = 2
    'jetId bit position'

    def _finalize(self, name, events):
        subjets = self.subjetmap.get(name, None)
        if subjets is not None and subjets in events.columns:
            subjets = events[subjets]
            embedded_subjets = type(subjets)(
                self._lazy_nested_crossref,
                args=([self._getcolumn('subJetIdx1'), self._getcolumn('subJetIdx2')], subjets),
                type=awkward.type.ArrayType(float('inf'), float('inf'), subjets.type.to.to),
            )
            embedded_subjets.__doc__ = subjets.__doc__
            self['subjets'] = embedded_subjets
            del self['subJetIdx1']
            del self['subJetIdx2']

    @property
    def isLoose(self):
        '''Returns a boolean array marking loose jets according to jetId index'''
        return (self.jetId & (1 << self.LOOSE)).astype(bool)

    @property
    def isTight(self):
        '''Returns a boolean array marking tight jets according to jetId index'''
        return (self.jetId & (1 << self.TIGHT)).astype(bool)

    @property
    def isTightLeptonVeto(self):
        '''Returns a boolean array marking tight jets with explicit lepton veto according to jetId index'''
        return (self.cutBasedBitmap & (1 << self.TIGHTLEPVETO)).astype(bool)


class Jet(LorentzVector):
    '''NanoAOD narrow radius jet object'''
    _enable_genjet = False
    'Set to true if using NanoAODv6 or newer (v5 had a bug in the mapping)'
    LOOSE = 0
    'jetId bit position'
    TIGHT = 1
    'jetId bit position'
    TIGHTLEPVETO = 2
    'jetId bit position'

    def _finalize(self, name, events):
        if 'Electron' in events.columns:
            electrons = events['Electron']
            reftype = awkward.type.ArrayType(float('inf'), float('inf'), electrons.type.to.to)
            reftype.check = False
            embedded_electrons = type(electrons)(
                self._lazy_nested_crossref,
                args=([self._getcolumn('electronIdx1'), self._getcolumn('electronIdx2')], electrons),
                type=reftype,
            )
            embedded_electrons.__doc__ = electrons.__doc__
            self['matched_electrons'] = embedded_electrons
            del self['electronIdx1']
            del self['electronIdx2']

        if 'Muon' in events.columns:
            muons = events['Muon']
            reftype = awkward.type.ArrayType(float('inf'), float('inf'), muons.type.to.to)
            reftype.check = False
            embedded_muons = type(muons)(
                self._lazy_nested_crossref,
                args=([self._getcolumn('muonIdx1'), self._getcolumn('muonIdx2')], muons),
                type=reftype,
            )
            embedded_muons.__doc__ = muons.__doc__
            self['matched_muons'] = embedded_muons
            del self['muonIdx1']
            del self['muonIdx2']

        if self._enable_genjet and 'GenJet' in events.columns:
            genjet = events['GenJet']
            reftype = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(genjet.type.to.to))
            reftype.check = False
            embedded_genjet = type(genjet)(
                self._lazy_crossref,
                args=(self._getcolumn('genJetIdx'), genjet),
                type=reftype,
            )
            embedded_genjet.__doc__ = genjet.__doc__
            self['matched_gen'] = embedded_genjet
            del self['genJetIdx']

        self.type.check = False

    @property
    def isLoose(self):
        '''Returns a boolean array marking loose jets according to jetId index'''
        return (self.jetId & (1 << self.LOOSE)).astype(bool)

    @property
    def isTight(self):
        '''Returns a boolean array marking tight jets according to jetId index'''
        return (self.jetId & (1 << self.TIGHT)).astype(bool)

    @property
    def isTightLeptonVeto(self):
        '''Returns a boolean array marking tight jets with explicit lepton veto according to jetId index'''
        return (self.cutBasedBitmap & (1 << self.TIGHTLEPVETO)).astype(bool)
