import numpy
import awkward1
from coffea.nanoevents.methods.mixin import mixin_class, mixin_method
from coffea.nanoevents.methods.base import NanoCollection
from coffea.nanoevents.methods.vector import LorentzVector, PtEtaPhiMLorentzVector


@mixin_class
class Candidate(LorentzVector):
    """A Lorentz vector with charge

    Properties this class requires: x, y, z, t, charge
    """

    @mixin_method(numpy.add, {"Candidate"})
    def add(self, other):
        return awkward1.zip(
            {
                "x": self.x + other.x,
                "y": self.y + other.y,
                "z": self.z + other.z,
                "t": self.t + other.t,
                "charge": self.charge + other.charge,
            },
            with_name="Candidate",
        )


@mixin_class
class PtEtaPhiMCandidate(Candidate, PtEtaPhiMLorentzVector):
    """A Lorentz vector in eta, mass coordinates with charge

    Properties this class requires: pt, eta, phi, mass, charge
    """

    pass


@mixin_class
class Photon(PtEtaPhiMCandidate, NanoCollection):
    @property
    def mass(self):
        return awkward1.broadcast_arrays(self.pt, 0.0)[1]

    @property
    def matched_gen(self):
        idx = self.genPartIdx.mask[self.genPartIdx >= 0]
        return self._events().GenPart[idx]


@mixin_class
class GenParticle(PtEtaPhiMLorentzVector, NanoCollection):
    @property
    def parent(self):
        idx = self.genPartIdxMother.mask[self.genPartIdxMother >= 0]
        return self._events().GenPart[idx]
