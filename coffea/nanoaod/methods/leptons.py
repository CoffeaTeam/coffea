import awkward
from .common import Candidate


class LeptonCommon(Candidate):
    def _finalize(self, name, events):
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

        self.type.check = False


class Electron(LeptonCommon):
    FAIL, VETO, LOOSE, MEDIUM, TIGHT = range(5)

    @property
    def isLoose(self):
        return (self.cutBased >= self.LOOSE).astype(bool)

    def _finalize(self, name, events):
        super(Electron, self)._finalize(name, events)
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


class Muon(LeptonCommon):
    def _finalize(self, name, events):
        super(Muon, self)._finalize(name, events)


class Photon(LeptonCommon):
    LOOSE, MEDIUM, TIGHT = range(3)

    def _finalize(self, name, events):
        del self['mass']

    @property
    def isLoose(self):
        return (self.cutBasedBitmap & (1 << self.LOOSE)).astype(bool)

    def _finalize(self, name, events):
        super(Photon, self)._finalize(name, events)
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


class Tau(LeptonCommon):
    def _finalize(self, name, events):
        super(Tau, self)._finalize(name, events)
