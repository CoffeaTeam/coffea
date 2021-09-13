from coffea.nanoevents import transforms
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms


class DelphesSchema(BaseSchema):
    """Delphes schema builder

    The Delphes schema is built from all branches found in the supplied file, based on
    the naming pattern of the branches. The following additional arrays are constructed:

    - Any branches named ``{name}_size`` are assumed to be counts branches and converted to offsets ``o{name}``
    """

    warn_missing_crossrefs = True

    mixins = {
        "CaloJet02": "ChargedParticle",
        "CaloJet04": "ChargedParticle",
        "CaloJet08": "ChargedParticle",
        "CaloJet15": "ChargedParticle",
        "EFlowNeutralHadron": "ChargedParticle",
        "EFlowPhoton": "Photon",
        "EFlowTrack": "ChargedParticle",
        "Electron": "Electron",
        "GenJet": "GenJet",
        "GenJet02": "ChargedParticle",
        "GenJet04": "ChargedParticle",
        "GenJet08": "ChargedParticle",
        "GenJet15": "ChargedParticle",
        "GenMissingET": "MissingET",
        "Jet": "Jet",
        "MissingET": "MissingET",
        "Muon": "Muon",
        "Particle": "GenParticle",
        "ParticleFlowJet02": "ChargedParticle",
        "ParticleFlowJet04": "ChargedParticle",
        "ParticleFlowJet08": "ChargedParticle",
        "ParticleFlowJet15": "ChargedParticle",
        "Photon": "Photon",
        # "ScalarHT": "",
        # pseudo-lorentz: pt, eta, phi, mass=0
        "Tower": "ChargedParticle",
        "Track": "ChargedParticle",
        "TrackJet02": "ChargedParticle",
        "TrackJet04": "ChargedParticle",
        "TrackJet08": "ChargedParticle",
        "TrackJet15": "ChargedParticle",
        "WeightLHEF": "ChargedParticle",
    }

    # These are stored as length-1 vectors unnecessarily
    singletons = ["Event", "EventLHEF", "HepMCEvent", "LHCOEvent"]

    def __init__(self, base_form, version="latest"):
        super().__init__(base_form)
        self._version = version
        if version == "latest":
            pass
        else:
            pass
        self._form["contents"] = self._build_collections(self._form["contents"])
        self._form["parameters"]["metadata"]["version"] = self._version

    @classmethod
    def v1(cls, base_form):
        """Build the DelphesEvents

        For example, one can use ``NanoEventsFactory.from_root("file.root", schemaclass=DelphesSchema.v1)``
        to ensure NanoAODv7 compatibility.
        """
        return cls(base_form, version="1")

    def _build_collections(self, branch_forms):
        # parse into high-level records (collections, list collections, and singletons)
        collections = set(k.split("/")[0] for k in branch_forms)
        collections -= set(k for k in collections if k.endswith("_size"))

        # Create offsets virtual arrays
        for name in collections:
            if f"{name}_size" in branch_forms:
                branch_forms[f"o{name}"] = transforms.counts2offsets_form(
                    branch_forms[f"{name}_size"]
                )

        output = {}
        for name in collections:
            mixin = self.mixins.get(name, "NanoCollection")
            # Every delphes collection is a list
            offsets = branch_forms["o" + name]
            content = {
                k[2 * len(name) + 2 :]: branch_forms[k]
                for k in branch_forms
                if k.startswith(name + "/" + name)
            }
            output[name] = zip_forms(
                content, name, record_name=mixin, offsets=offsets
            )
            output[name]["content"]["parameters"].update(
                {
                    "__doc__": offsets["parameters"]["__doc__"],
                    "collection_name": name,
                }
            )
            if name in self.singletons:
                # flatten!
                output[name] = output[name]["content"]
        return output

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import delphes

        return delphes.behavior
