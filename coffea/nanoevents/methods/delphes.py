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
        return f"<Delphes event {self.Event.Number}>"


behavior["NanoEvents"] = DelphesEvents


def _set_repr_name(classname):
    def namefcn(self):
        return classname

    behavior[("__typestr__", classname)] = classname[0].lower() + classname[1:]
    behavior[classname].__repr__ = namefcn


@awkward.mixin_class(behavior)
class Event:
    ...


_set_repr_name("Event")


@awkward.mixin_class(behavior)
class LHEFEvent(Event):
    ...


_set_repr_name("LHEFEvent")


@awkward.mixin_class(behavior)
class HepMCEvent(Event):
    ...


_set_repr_name("HepMCEvent")


@awkward.mixin_class(behavior)
class LHCOEvent(Event):
    ...


_set_repr_name("LHCOEvent")


@awkward.mixin_class(behavior)
class Particle(vector.PtEtaPhiMLorentzVector):
    """Generic particle collection that has Lorentz vector properties

    The following branches are not used:

     - E: particle energy
     - Px: particle momentum vector (x component)
     - Py: particle momentum vector (y component)
     - Pz: particle momentum vector (z component)

     - P: particle momentum
     - Rapidity: particle rapidity

     - T: particle vertex position (t component)
     - X: particle vertex position (x component)
     - Y: particle vertex position (y component)
     - Z: particle vertex position (z component)
    """

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
class Vertex(vector.LorentzVector):
    """Generic vertex collection that has Lorentz vector properties"""

    @property
    def t(self):
        return self["T"]

    @property
    def x(self):
        return self["X"]

    @property
    def y(self):
        return self["Y"]

    @property
    def z(self):
        return self["Z"]


@awkward.mixin_class(behavior)
class Electron(Particle, base.NanoCollection):
    ...


_set_repr_name("Electron")


@awkward.mixin_class(behavior)
class Muon(Particle, base.NanoCollection):
    ...


_set_repr_name("Muon")


@awkward.mixin_class(behavior)
class Photon(Particle, base.NanoCollection):
    ...


_set_repr_name("Photon")


@awkward.mixin_class(behavior)
class Jet(Particle, base.NanoCollection):
    ...


_set_repr_name("Jet")


@awkward.mixin_class(behavior)
class GenJet(Particle, base.NanoCollection):
    ...


_set_repr_name("GenJet")


@awkward.mixin_class(behavior)
class Particle(Particle, base.NanoCollection):
    ...


_set_repr_name("Particle")


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
    "Event",
    "LHEFEvent",
    "HepMCEvent",
    "Particle",
    "Electron",
    "Muon",
    "Photon",
    "Jet",
    "GenJet",
    "MissingET",
]
