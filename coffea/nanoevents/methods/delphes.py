"""Mixins for the Delphes schema

See https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/RootTreeDescription for details.
"""
import awkward
from coffea.nanoevents.methods import base, vector, candidate

behavior = {}
behavior.update(base.behavior)
# vector behavior is included in candidate behavior
behavior.update(candidate.behavior)


class DelphesEvents(behavior["NanoEvents"]):
    def __repr__(self):
        return "<Delphes event>"


behavior["NanoEvents"] = DelphesEvents


def _set_repr_name(classname):
    def namefcn(self):
        return classname

    behavior[("__typestr__", classname)] = classname[0].lower() + classname[1:]
    behavior[classname].__repr__ = namefcn


@awkward.mixin_class(behavior)
class Particle(vector.PtEtaPhiMLorentzVector):
    """Generic particle collection that has Lorentz vector properties"""

    @property
    def pt(self):
        return self["PT"]

    @property
    def eta(self):
        return self["Eta"]

    @property
    def phi(self):
        return self["Phi"]

    @property
    def mass(self):
        return self["Mass"]


_set_repr_name("Particle")


@awkward.mixin_class(behavior)
class ChargedParticle(Particle):
    @property
    def charge(self):
        return self["Charge"]


_set_repr_name("ChargedParticle")


@awkward.mixin_class(behavior)
class Electron(ChargedParticle, base.NanoCollection):
    ...


_set_repr_name("Electron")


@awkward.mixin_class(behavior)
class Muon(ChargedParticle, base.NanoCollection):
    ...


_set_repr_name("Muon")


@awkward.mixin_class(behavior)
class Photon(ChargedParticle, base.NanoCollection):
    ...


_set_repr_name("Photon")


@awkward.mixin_class(behavior)
class Jet(ChargedParticle, base.NanoCollection):
    ...


_set_repr_name("Jet")


@awkward.mixin_class(behavior)
class GenJet(ChargedParticle, base.NanoCollection):
    ...


_set_repr_name("GenJet")


@awkward.mixin_class(behavior)
class GenParticle(ChargedParticle, base.NanoCollection):
    ...


_set_repr_name("GenParticle")


@awkward.mixin_class(behavior)
class MissingET(vector.PolarTwoVector, base.NanoCollection):
    @property
    def r(self):
        return self["MET"]

    @property
    def phi(self):
        return self["Phi"]


_set_repr_name("MissingET")

__all__ = [
    "DelphesEvents",
    "Particle",
    "ChargedParticle",
    "Electron",
    "Muon",
    "Photon",
    "Jet",
    "GenJet",
    "MissingET",
    "GenParticle",
]
