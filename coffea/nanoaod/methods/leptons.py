from .common import Candidate


class Electron(Candidate):
    FAIL, VETO, LOOSE, MEDIUM, TIGHT = range(5)

    @property
    def isLoose(self):
        return (self.cutBased >= self.LOOSE).astype(bool)


class Muon(Candidate):
    pass


class Photon(Candidate):
    LOOSE, MEDIUM, TIGHT = range(3)

    @property
    def isLoose(self):
        return (self.cutBasedBitmap & (1 << self.LOOSE)).astype(bool)


class Tau(Candidate):
    pass
