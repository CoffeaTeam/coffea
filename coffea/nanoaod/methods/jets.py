from .common import LorentzVector
import awkward


class FatJet(LorentzVector):
    '''NanoAOD large radius jet object'''
    enable_genjet = False
    'Set True if FatJet_genJetAK8Idx is available'
    subjetmap = {'FatJet': 'SubJet'}  # V6 has 'GenJetAK8': 'SubGenJetAK8', maybe better to put in generator.py
    'If additional large-radius jet collections are available, add here their associated subjet collections'
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

        if self.enable_genjet and 'GenJetAK8' in events.columns:
            genjet = events['GenJetAK8']
            reftype = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(genjet.type.to.to))
            reftype.check = False
            embedded_genjet = type(genjet)(
                self._lazy_crossref,
                args=(self._getcolumn('genJetAK8Idx'), genjet),
                type=reftype,
            )
            embedded_genjet.__doc__ = genjet.__doc__
            self['matched_gen'] = embedded_genjet
            del self['genJetAK8Idx']

        if 'FatJetPFCands' in events.columns:
            pfcand_link = events['FatJetPFCands']
            reftype = awkward.type.ArrayType(float('inf'), float('inf'), pfcand_link.type.to.to)
            reftype.check = False
            embedded_pfcand_extras = type(pfcand_link)(
                self._lazy_double_jagged,
                args=(self._getcolumn('nConstituents'), pfcand_link),
                type=reftype,
            )
            embedded_pfcand_extras.__doc__ = pfcand_link.__doc__
            self['constituentExtras'] = embedded_pfcand_extras

            pfcands = events['PFCands']
            indexedtype = awkward.type.ArrayType(float('inf'), pfcands.type.to.to)
            indexedtype.check = False
            jetpfcands = type(pfcands)(
                pfcand_link._lazy_crossref,
                args=(pfcand_link._getcolumn('candIdx'), pfcands),
                type=indexedtype,
            )
            reftype = awkward.type.ArrayType(float('inf'), float('inf'), pfcands.type.to.to)
            reftype.check = False
            embedded_pfcands = type(pfcands)(
                self._lazy_double_jagged,
                args=(self._getcolumn('nConstituents'), jetpfcands),
                type=reftype,
            )
            embedded_pfcands.__doc__ = pfcands.__doc__
            self['constituents'] = embedded_pfcands

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
        return (self.jetId & (1 << self.TIGHTLEPVETO)).astype(bool)


class Jet(LorentzVector):
    '''NanoAOD narrow radius jet object'''
    enable_genjet = True
    'Set False for NanoAODv5, which had a bug with the mapping betwen fatjets and genjets'
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

        if self.enable_genjet and 'GenJet' in events.columns:
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

        if 'JetPFCands' in events.columns:
            pfcand_link = events['JetPFCands']
            reftype = awkward.type.ArrayType(float('inf'), float('inf'), pfcand_link.type.to.to)
            reftype.check = False
            embedded_pfcand_extras = type(pfcand_link)(
                self._lazy_double_jagged,
                args=(self._getcolumn('nConstituents'), pfcand_link),
                type=reftype,
            )
            embedded_pfcand_extras.__doc__ = pfcand_link.__doc__
            self['constituentExtras'] = embedded_pfcand_extras

            pfcands = events['PFCands']
            indexedtype = awkward.type.ArrayType(float('inf'), pfcands.type.to.to)
            indexedtype.check = False
            jetpfcands = type(pfcands)(
                pfcand_link._lazy_crossref,
                args=(pfcand_link._getcolumn('candIdx'), pfcands),
                type=indexedtype,
            )
            reftype = awkward.type.ArrayType(float('inf'), float('inf'), pfcands.type.to.to)
            reftype.check = False
            embedded_pfcands = type(pfcands)(
                self._lazy_double_jagged,
                args=(self._getcolumn('nConstituents'), jetpfcands),
                type=reftype,
            )
            embedded_pfcands.__doc__ = pfcands.__doc__
            self['constituents'] = embedded_pfcands

        # disable this type check due to cyclic reference through leptons
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
        return (self.jetId & (1 << self.TIGHTLEPVETO)).astype(bool)
