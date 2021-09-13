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

    multi_mixins = {"Event": ["Event", "LHEFEvent", "HepMCEvent", "LHCOEvent"]}

    """Default configuration for mixin types, based on the collection name.

    The types are implemented in the `coffea.nanoevents.methods.nanoaod` module.
    """

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
            multi_mixin = self.multi_mixins.get(name)
            if "o" + name in branch_forms and name not in branch_forms:
                # list collection
                offsets = branch_forms["o" + name]
                content = {
                    k[2 * len(name) + 2 :]: branch_forms[k]
                    for k in branch_forms
                    if k.startswith(name + "/" + name)
                }
                if multi_mixin is None:
                    output[name] = zip_forms(
                        content, name, record_name=mixin, offsets=offsets
                    )
                    output[name]["content"]["parameters"].update(
                        {
                            "__doc__": offsets["parameters"]["__doc__"],
                            "collection_name": name,
                        }
                    )
                else:
                    for mixin_name in multi_mixin:
                        # the output[name] will create `self.{name}` correspondingly
                        output[mixin_name] = zip_forms(
                            content, name, record_name=mixin_name, offsets=offsets
                        )
                        output[mixin_name]["content"]["parameters"].update(
                            {
                                "__doc__": offsets["parameters"]["__doc__"],
                                "collection_name": name,
                            }
                        )

            elif "o" + name in branch_forms:
                # list singleton, can use branch's own offsets
                output[name] = branch_forms[name]
                output[name].setdefault("parameters", {})
                output[name]["parameters"].update(
                    {"__array__": mixin, "collection_name": name}
                )
            elif name in branch_forms:
                # singleton
                output[name] = branch_forms[name]
            else:
                # simple collection
                output[name] = zip_forms(
                    {
                        k[len(name) + 1 :]: branch_forms[k]
                        for k in branch_forms
                        if k.startswith(name + "_")
                    },
                    name,
                    record_name=mixin,
                )
                output[name].setdefault("parameters", {})
                output[name]["parameters"].update({"collection_name": name})

        return output

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import delphes

        return delphes.behavior
