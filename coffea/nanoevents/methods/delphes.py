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
    @property
    def number(self):
        """event number"""
        return self["Number"]

    @property
    def read_time(self):
        """read time"""
        return self["ReadTime"]

    @property
    def proc_time(self):
        """processing time"""
        return self["ProcTime"]


_set_repr_name("Event")


@awkward.mixin_class(behavior)
class LHEFEvent(Event):
    @property
    def process_id(self):
        """subprocess code for the event"""
        return self["ProcessID"]

    @property
    def weight(self):
        """weight for the event"""
        return self["Weight"]

    @property
    def xsec(self):
        """cross-section"""
        return self["CrossSection"]

    @property
    def scale_pdf(self):
        """scale in GeV used in the calculation of the PDFs in the event"""
        return self["ScalePDF"]

    @property
    def alpha_qed(self):
        """value of the QED coupling used in the event"""
        return self["AlphaQED"]

    @property
    def alpha_qcd(self):
        """value of the QCD coupling used in the event"""
        return self["AlphaQCD"]


_set_repr_name("LHEFEvent")


@awkward.mixin_class(behavior)
class HepMCEvent(Event):
    @property
    def process_id(self):
        """subprocess code for the event"""
        return self["ProcessID"]

    @property
    def mu(self):
        """number of multi parton interactions"""
        return self["MPI"]

    @property
    def weight(self):
        """weight for the event"""
        return self["Weight"]

    @property
    def xsec(self):
        """cross-section in pb"""
        return self["CrossSection"]

    @property
    def xsec_err(self):
        """cross-section error in pb"""
        return self["CrossSectionError"]

    @property
    def scale(self):
        """energy scale, see hep-ph/0109068"""
        return self["Scale"]

    @property
    def alpha_qed(self):
        """QED coupling, see hep-ph/0109068"""
        return self["AlphaQED"]

    @property
    def alpha_qcd(self):
        """QCD coupling, see hep-ph/0109068"""
        return self["AlphaQCD"]

    @property
    def parton1_id(self):
        """flavour code of first parton"""
        return self["ID1"]

    @property
    def parton2_id(self):
        """flavour code of second parton"""
        return self["ID2"]

    @property
    def parton1_efrac(self):
        """fraction of beam momentum carried by first parton ("beam side")"""
        return self["X1"]

    @property
    def parton2_efrac(self):
        """fraction of beam momentum carried by second parton ("target side")"""
        return self["X2"]

    @property
    def scale_pdf(self):
        """Q-scale used in evaluation of PDF's (in GeV)"""
        return self["ScalePDF"]

    @property
    def pdf1(self):
        """PDF (id1, x1, Q)"""
        return self["PDF1"]

    @property
    def pdf2(self):
        """PDF (id2, x2, Q)"""
        return self["PDF2"]


_set_repr_name("HepMCEvent")


@awkward.mixin_class(behavior)
class LHCOEvent(Event):
    @property
    def trigger(self):
        """trigger word"""
        return self["Trigger"]


_set_repr_name("LHCOEvent")


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
    "Event",
    "LHEFEvent",
    "HepMCEvent",
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
