"""Mixins for the Delphes schema

See https://cp3.irmp.ucl.ac.be/projects/delphes/wiki/WorkBook/RootTreeDescription for details.
"""

import awkward
import numpy

from coffea.nanoevents.methods import base, candidate, vector

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
class Event: ...


_set_repr_name("Event")


@awkward.mixin_class(behavior)
class LHEFEvent(Event): ...


_set_repr_name("LHEFEvent")


@awkward.mixin_class(behavior)
class HepMCEvent(Event): ...


_set_repr_name("HepMCEvent")


@awkward.mixin_class(behavior)
class LHCOEvent(Event): ...


_set_repr_name("LHCOEvent")


@awkward.mixin_class(behavior)
class Weight(base.NanoCollection): ...


_set_repr_name("Weight")


@awkward.mixin_class(behavior)
class WeightLHEF(Event): ...


_set_repr_name("WeightLHEF")


@awkward.mixin_class(behavior)
class Rho(base.NanoCollection): ...


_set_repr_name("Rho")


@awkward.mixin_class(behavior)
class ScalarHT(base.NanoCollection): ...


_set_repr_name("ScalarHT")


@awkward.mixin_class(behavior)
class MissingET(vector.SphericalThreeVector, base.NanoCollection):
    @property
    def rho(self):
        return self["MET"] * numpy.cosh(self.eta)

    @property
    def theta(self):
        return 2 * numpy.arctan(numpy.exp(-self.eta))

    @property
    def phi(self):
        return self["Phi"]

    @property
    def eta(self):
        return self["Eta"]


_set_repr_name("MissingET")


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


_set_repr_name("Vertex")


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
class MasslessParticle(Particle, base.NanoCollection):
    @property
    def mass(self):
        return 0.0 * self.pt


_set_repr_name("MasslessParticle")


@awkward.mixin_class(behavior)
class Photon(MasslessParticle, base.NanoCollection): ...


_set_repr_name("Photon")


@awkward.mixin_class(behavior)
class Electron(MasslessParticle, base.NanoCollection): ...


_set_repr_name("Electron")


@awkward.mixin_class(behavior)
class Muon(MasslessParticle, base.NanoCollection): ...


_set_repr_name("Muon")


@awkward.mixin_class(behavior)
class Jet(Particle, base.NanoCollection): ...


_set_repr_name("Jet")


@awkward.mixin_class(behavior)
class Track(Particle, base.NanoCollection): ...


_set_repr_name("Track")


@awkward.mixin_class(behavior)
class Tower(MasslessParticle, base.NanoCollection):
    @property
    def pt(self):
        return self["ET"]


_set_repr_name("Tower")


__all__ = [
    "DelphesEvents",
    "Event",
    "LHEFEvent",
    "HepMCEvent",
    "LHCOEvent",
    "Weight",
    "WeightLHEF",
    "Rho",
    "ScalarHT",
    "MissingET",
    "Vertex",
    "Particle",
    "Photon",
    "Electron",
    "Muon",
    "Jet",
    "Track",
    "Tower",
]
