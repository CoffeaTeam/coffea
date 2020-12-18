"""Physics object candidate mixin

This provides just a Lorentz vector with charge, but maybe
in the future it will provide some sort of composite candiate building tool
that automatically resolves duplicates in the chain.
"""
import numpy
import awkward as ak
from coffea.nanoevents.methods import vector


behavior = dict(vector.behavior)


@ak.mixin_class(behavior)
class Candidate(vector.LorentzVector):
    """A Lorentz vector with charge

    This mixin class requires the parent class to provide items `x`, `y`, `z`, `t`, and `charge`.
    """

    @ak.mixin_class_method(numpy.add, {"Candidate"})
    def add(self, other):
        """Add two candidates together elementwise using `x`, `y`, `z`, `t`, and `charge` components"""
        return ak.zip(
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
        return ak.zip(
            {
                "x": ak.sum(self.x, axis=axis),
                "y": ak.sum(self.y, axis=axis),
                "z": ak.sum(self.z, axis=axis),
                "t": ak.sum(self.t, axis=axis),
                "charge": ak.sum(self.charge, axis=axis),
            },
            with_name="Candidate",
        )


@ak.mixin_class(behavior)
class PtEtaPhiMCandidate(Candidate, vector.PtEtaPhiMLorentzVector):
    """A Lorentz vector in eta, mass coordinates with charge

    This mixin class requires the parent class to provide items `x`, `y`, `z`, `t`, and `charge`.
    """

    pass


__all__ = ["Candidate", "PtEtaPhiMCandidate"]
