import numpy
import awkward1
from coffea.nanoevents.methods.mixin import mixin_class, mixin_method
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

    def sum(self, axis=-1):
        return awkward1.zip(
            {
                "x": awkward1.sum(self.x, axis=axis),
                "y": awkward1.sum(self.y, axis=axis),
                "z": awkward1.sum(self.z, axis=axis),
                "t": awkward1.sum(self.t, axis=axis),
                "charge": awkward1.sum(self.charge, axis=axis),
            },
            with_name="Candidate",
        )


@mixin_class
class PtEtaPhiMCandidate(Candidate, PtEtaPhiMLorentzVector):
    """A Lorentz vector in eta, mass coordinates with charge

    Properties this class requires: pt, eta, phi, mass, charge
    """

    pass
