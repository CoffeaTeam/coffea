import awkward
from .common import Candidate


class LeptonCommon(Candidate):
    '''Embeds all common gen particle and jet cross-references'''
    def _finalize(self, name, events):
        if 'Jet' in events.columns:
            jets = events['Jet']
            reftype = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(jets.type.to.to))
            reftype.check = False
            embedded_jet = type(jets)(
                self._lazy_crossref,
                args=(self._getcolumn('jetIdx'), jets),
                type=reftype,
            )
            embedded_jet.__doc__ = jets.__doc__
            self['matched_jet'] = embedded_jet
            del self['jetIdx']

        if 'GenPart' in events.columns:
            genpart = events['GenPart']
            reftype = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(genpart.type.to.to))
            reftype.check = False
            embedded_genpart = type(genpart)(
                self._lazy_crossref,
                args=(self._getcolumn('genPartIdx'), genpart),
                type=reftype,
            )
            embedded_genpart.__doc__ = genpart.__doc__
            self['matched_gen'] = embedded_genpart
            del self['genPartIdx']

        # disable this type check due to cyclic reference through jets
        self.type.check = False


class Electron(LeptonCommon):
    '''NanoAOD electron object'''
    FAIL = 0
    'cutBased selection minimum value'
    VETO = 1
    'cutBased selection minimum value'
    LOOSE = 2
    'cutBased selection minimum value'
    MEDIUM = 3
    'cutBased selection minimum value'
    TIGHT = 4
    'cutBased selection minimum value'

    def _finalize(self, name, events):
        super(Electron, self)._finalize(name, events)
        if 'Photon' in events.columns:
            photons = events['Photon']
            reftype = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(photons.type.to.to))
            reftype.check = False
            embedded_photon = type(photons)(
                self._lazy_crossref,
                args=(self._getcolumn('photonIdx'), photons),
                type=reftype,
            )
            embedded_photon.__doc__ = photons.__doc__
            self['matched_photon'] = embedded_photon
            del self['photonIdx']

    @property
    def isVeto(self):
        '''Returns a boolean array marking veto-level cut-based photons'''
        return (self.cutBased & (1 << self.VETO)).astype(bool)

    @property
    def isLoose(self):
        '''Returns a boolean array marking loose cut-based photons'''
        return (self.cutBased & (1 << self.LOOSE)).astype(bool)

    @property
    def isMedium(self):
        '''Returns a boolean array marking medium cut-based photons'''
        return (self.cutBased & (1 << self.MEDIUM)).astype(bool)

    @property
    def isTight(self):
        '''Returns a boolean array marking tight cut-based photons'''
        return (self.cutBased & (1 << self.TIGHT)).astype(bool)


class Muon(LeptonCommon):
    '''NanoAOD muon object'''
    def _finalize(self, name, events):
        super(Muon, self)._finalize(name, events)
        if 'FsrPhoton' in events.columns:
            photons = events['FsrPhoton']
            reftype = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(photons.type.to.to))
            reftype.check = False
            embedded_photon = type(photons)(
                self._lazy_crossref,
                args=(self._getcolumn('fsrPhotonIdx'), photons),
                type=reftype,
            )
            embedded_photon.__doc__ = photons.__doc__
            self['matched_fsrPhoton'] = embedded_photon
            del self['fsrPhotonIdx']


class Photon(LeptonCommon):
    '''NanoAOD photon object'''
    LOOSE = 0
    'cutBasedBitmap bit position'
    MEDIUM = 1
    'cutBasedBitmap bit position'
    TIGHT = 2
    'cutBasedBitmap bit position'

    @property
    def isLoose(self):
        '''Returns a boolean array marking loose cut-based photons'''
        return (self.cutBasedBitmap & (1 << self.LOOSE)).astype(bool)

    @property
    def isMedium(self):
        '''Returns a boolean array marking medium cut-based photons'''
        return (self.cutBasedBitmap & (1 << self.MEDIUM)).astype(bool)

    @property
    def isTight(self):
        '''Returns a boolean array marking tight cut-based photons'''
        return (self.cutBasedBitmap & (1 << self.TIGHT)).astype(bool)

    def _finalize(self, name, events):
        del self['mass']
        super(Photon, self)._finalize(name, events)
        if 'Electron' in events.columns:
            electrons = events['Electron']
            reftype = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(electrons.type.to.to))
            reftype.check = False
            embedded_electron = type(electrons)(
                self._lazy_crossref,
                args=(self._getcolumn('electronIdx'), electrons),
                type=reftype,
            )
            embedded_electron.__doc__ = electrons.__doc__
            self['matched_electron'] = embedded_electron
            del self['electronIdx']


class FsrPhoton(Candidate):
    '''NanoAOD fsr photon object'''

    def _finalize(self, name, events):
        del self['mass']
        if 'Muon' in events.columns:
            muons = events['Muon']
            reftype = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(muons.type.to.to))
            reftype.check = False
            embedded_muon = type(muons)(
                self._lazy_crossref,
                args=(self._getcolumn('muonIdx'), muons),
                type=reftype,
            )
            embedded_muon.__doc__ = muons.__doc__
            self['matched_muon'] = embedded_muon
            del self['muonIdx']

        # disable this type check due to cyclic reference through jets
        self.type.check = False


class Tau(LeptonCommon):
    '''NanoAOD tau object'''
    def _finalize(self, name, events):
        super(Tau, self)._finalize(name, events)
