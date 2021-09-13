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
        "CaloJet02": "Particle",
        "CaloJet04": "Particle",
        "CaloJet08": "Particle",
        "CaloJet15": "Particle",
        "EFlowNeutralHadron": "Particle",
        "EFlowPhoton": "Photon",
        "EFlowTrack": "Particle",
        "Electron": "Electron",
        "GenJet": "GenJet",
        "GenJet02": "Particle",
        "GenJet04": "Particle",
        "GenJet08": "Particle",
        "GenJet15": "Particle",
        "GenMissingET": "MissingET",
        "Jet": "Jet",
        "MissingET": "MissingET",
        "Muon": "Muon",
        "Particle": "Particle",
        "ParticleFlowJet02": "Particle",
        "ParticleFlowJet04": "Particle",
        "ParticleFlowJet08": "Particle",
        "ParticleFlowJet15": "Particle",
        "Photon": "Photon",
        # "ScalarHT": "",
        # pseudo-lorentz: pt, eta, phi, mass=0
        "Tower": "Particle",
        "Track": "Particle",
        "TrackJet02": "Particle",
        "TrackJet04": "Particle",
        "TrackJet08": "Particle",
        "TrackJet15": "Particle",
        "WeightLHEF": "Particle",
    }

    # These are stored as length-1 vectors unnecessarily
    singletons = ["Event", "EventLHEF", "HepMCEvent", "LHCOEvent"]

    docstrings = {
        "Number": "event number",
        "ReadTime": "read time",
        "ProcTime": "processing time",
        "ProcessID": "subprocess code for the event",
        "Weight": "weight for the event",
        "CrossSection": "cross-section in [pb]",
        "CrossSectionError": "cross-section error [pb]",
        "ScalePDF": "Q-scale used in evaluation of PDF's [GeV]",
        "AlphaQED": "value of the QED coupling used in the event, see hep-ph/0109068",
        "AlphaQCD": "value of the QCD coupling used in the event, see hep-ph/0109068",
        "MPI": "number of multi parton interactions",
        "Scale": "energy scale, see hep-ph/0109068",
        "ID1": "flavour code of first parton",
        "ID2": "flavour code of second parton",
        "X1": 'fraction of beam momentum carried by first parton ("beam side")',
        "X2": 'fraction of beam momentum carried by second parton ("target side")',
        "PDF1": "PDF (id1, x1, Q)",
        "PDF2": "PDF (id2, x2, Q)",
        "Trigger": "trigger word",
        "PID": "particle HEP ID number",
        "Status": "particle status",
        "IsPU": "0 or 1 for particles from pile-up interactions",
        "M1": "particle first parent",
        "M2": "particle second parent",
        "D1": "particle first child",
        "D2": "particle last child",
    }

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
            output[name] = zip_forms(content, name, record_name=mixin, offsets=offsets)
            output[name]["content"]["parameters"].update(
                {
                    "__doc__": offsets["parameters"]["__doc__"],
                    "collection_name": name,
                }
            )

            # update docstrings as needed
            # NB: must be before flattening for easier logic
            for parameter in output[name]["content"]["contents"].keys():
                output[name]["content"]["contents"][parameter]["parameters"][
                    "__doc__"
                ] = self.docstrings.get(
                    parameter,
                    output[name]["content"]["contents"][parameter]["parameters"][
                        "__doc__"
                    ],
                )

            if name in self.singletons:
                # flatten! this 'promotes' the content of an inner dimension
                # upwards, effectively hiding one nested dimension
                output[name] = output[name]["content"]

        return output

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import delphes

        return delphes.behavior
