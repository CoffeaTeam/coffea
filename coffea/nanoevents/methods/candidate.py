"""Physics object candidate mixin

This provides just a Lorentz vector with charge, but maybe
in the future it will provide some sort of composite candiate building tool
that automatically resolves duplicates in the chain.
"""
import numpy
import awkward
from coffea.nanoevents.methods import vector


behavior = dict(vector.behavior)


@awkward.mixin_class(behavior)
class Candidate(vector.LorentzVector):
    """A Lorentz vector with charge

    This mixin class requires the parent class to provide items `x`, `y`, `z`, `t`, and `charge`.
    """

    @awkward.mixin_class_method(numpy.add, {"Candidate"})
    def add(self, other):
        """Add two candidates together elementwise using `x`, `y`, `z`, `t`, and `charge` components"""
        return awkward.zip(
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
        """Sum an array of vectors elementwise using `x`, `y`, `z`, `t`, and `charge` components"""
        return awkward.zip(
            {
                "x": awkward.sum(self.x, axis=axis),
                "y": awkward.sum(self.y, axis=axis),
                "z": awkward.sum(self.z, axis=axis),
                "t": awkward.sum(self.t, axis=axis),
                "charge": awkward.sum(self.charge, axis=axis),
            },
            with_name="Candidate",
        )


@awkward.mixin_class(behavior)
class PtEtaPhiMCandidate(Candidate, vector.PtEtaPhiMLorentzVector):
    """A Lorentz vector in eta, mass coordinates with charge

    This mixin class requires the parent class to provide items `pt`, `eta`, `phi`, `mass`, and `charge`.
    """

    pass


@awkward.mixin_class(behavior)
class PtEtaPhiECandidate(Candidate, vector.PtEtaPhiELorentzVector):
    """A Lorentz vector in eta, energy coordinates with charge

    This mixin class requires the parent class to provide items `pt`, `eta`, `phi`, `energy`, and `charge`.
    """

    pass


__all__ = ["Candidate", "PtEtaPhiMCandidate", "PtEtaPhiECandidate"]
