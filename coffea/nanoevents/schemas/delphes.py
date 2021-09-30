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
        "CaloJet02": "Jet",
        "CaloJet04": "Jet",
        "CaloJet08": "Jet",
        "CaloJet15": "Jet",
        "EFlowNeutralHadron": "Tower",
        "EFlowPhoton": "Photon",
        "EFlowTrack": "Track",
        "Electron": "Electron",
        "GenJet": "Jet",
        "GenJet02": "Jet",
        "GenJet04": "Jet",
        "GenJet08": "Jet",
        "GenJet15": "Jet",
        "GenMissingET": "MissingET",
        "Jet": "Jet",
        "MissingET": "MissingET",
        "Muon": "Muon",
        "Particle": "Particle",
        "ParticleFlowJet02": "Jet",
        "ParticleFlowJet04": "Jet",
        "ParticleFlowJet08": "Jet",
        "ParticleFlowJet15": "Jet",
        "Photon": "Photon",
        "Tower": "Tower",
        "Track": "Track",
        "TrackJet02": "Jet",
        "TrackJet04": "Jet",
        "TrackJet08": "Jet",
        "TrackJet15": "Jet",
        "Weight": "Weight",
        "WeightLHEF": "WeightLHEF",
        # the following are also singletons
        "Event": "Event",
        "EventLHEF": "EventLHEF",
        "HepMCEvent": "HepMCEvent",
        "LHCOEvent": "LHCOEvent",
        "Rho": "Rho",
        "ScalarHT": "ScalarHT",
        "MissingET": "MissingET",
    }

    # These are stored as length-1 vectors unnecessarily
    singletons = [
        "Event",
        "EventLHEF",
        "HepMCEvent",
        "LHCOEvent",
        "Rho",
        "ScalarHT",
        "MissingET",
    ]

    docstrings = {
        "AlphaQCD": "value of the QCD coupling used in the event, see hep-ph/0109068",
        "AlphaQED": "value of the QED coupling used in the event, see hep-ph/0109068",
        "Area": "area",
        "BTVSumPT2": "sum pt^2 of tracks attached to the secondary vertex",
        "BTag": "0 or 1 for a jet that has been tagged as containing a heavy quark",
        "BTagAlgo": "0 or 1 for a jet that has been tagged as containing a heavy quark",
        "BTagPhys": "0 or 1 for a jet that has been tagged as containing a heavy quark",
        "Beta": "(sum pt of charged pile-up constituents)/(sum pt of charged constituents)",
        "BetaStar": "(sum pt of charged constituents coming from hard interaction)/(sum pt of charged constituents)",
        "Charge": "charge",
        "Constituents": "references to constituents",
        "Constituents": "references to constituents",
        "CrossSection": "cross-section in [pb]",
        "CrossSectionError": "cross-section error [pb]",
        "CtgTheta": "cotangent of theta",
        "D0": "transverse impact parameter",
        "D1": "particle first child",
        "D2": "particle last child",
        "DZ": "longitudinal impact parameter",
        "E": "energy [GeV]",
        "ET": "transverse energy [GeV]",
        "Edges[2]": "pseudorapidity range edges",
        "Edges[4]": "calorimeter tower edges",
        "Eem": "calorimeter tower electromagnetic energy",
        "Ehad": "calorimeter tower hadronic energy",
        "EhadOverEem": "ratio of the hadronic versus electromagnetic energy deposited in the calorimeter",
        "ErrorCtgTheta": "cotangent of theta error",
        "ErrorD0": "transverse impact parameter error",
        "ErrorDZ": "longitudinal impact parameter error",
        "ErrorP": "momentum error [GeV]",
        "ErrorPT": "transverse momentum error [GeV]",
        "ErrorPhi": "azimuthal angle error",
        "ErrorT": "vertex position error (t component)",
        "ErrorX": "vertex position error (x component)",
        "ErrorY": "vertex position error (y component)",
        "ErrorZ": "vertex position error (z component)",
        "Eta": "pseudorapidity",
        "EtaOuter": "pseudorapidity at the edge",
        "Flavor": "jet flavor",
        "FlavorAlgo": "jet flavor",
        "FlavorPhys": "jet flavor",
        "FracPt[5]": "(sum pt of constituents within a ring 0.1*i < DeltaR < 0.1*(i+1))/(sum pt of constituents)",
        "GenDeltaZ": "distance in z to closest generated vertex",
        "GenSumPT2": "sum pt^2 of gen tracks attached to the vertex",
        "HT": "scalar sum of transverse momenta [GeV]",
        "ID": "ID",
        "ID1": "flavour code of first parton",
        "ID2": "flavour code of second parton",
        "Index": "index",
        "IsPU": "0 or 1 for particles from pile-up interactions",
        "IsolationVar": "isolation variable",
        "IsolationVarRhoCorr": "isolation variable",
        "L": "path length",
        "M1": "particle first parent",
        "M2": "particle second parent",
        "MET": "missing transverse energy",
        "MPI": "number of multi parton interactions",
        "Mass": "invariant mass [GeV]",
        "MeanSqDeltaR": "average distance (squared) between constituent and particle weighted by pt (squared) of constituent",
        "NCharged": "number of charged constituents",
        "NDF": "number of degrees of freedom",
        "NNeutrals": "number of neutral constituents",
        "NSubJetsPruned": "number of subjets pruned",
        "NSubJetsSoftDropped": "number of subjets soft-dropped",
        "NSubJetsTrimmed": "number of subjets trimmed",
        "NTimeHits": "number of hits contributing to time measurement",
        "Number": "event number",
        "P": "momentum [GeV]",
        "PDF1": "PDF (id1, x1, Q)",
        "PDF2": "PDF (id2, x2, Q)",
        "PID": "HEP ID number",
        "PT": "transverse momentum [GeV]",
        "PTD": "average pt between constituent and jet weighted by pt of constituent",
        "Particle": "reference to generated particle",
        "Particles": "references to generated particles",
        "Phi": "azimuthal angle",
        "PhiOuter": "azimuthal angle at the edge",
        "ProcTime": "processing time",
        "ProcessID": "subprocess code for the event or signal process id",
        "PrunedP4[5]": "first entry (i = 0) is the total Pruned Jet 4-momenta and from i = 1 to 4 are the pruned subjets 4-momenta",
        "Px": "particle momentum vector (x component)",
        "Py": "particle momentum vector (y component)",
        "Pz": "particle momentum vector (z component)",
        "Rapidity": "particle rapidity",
        "ReadTime": "read time",
        "Rho": "rho energy density",
        "S": "distance to the interaction point [m]",
        "Scale": "energy scale, see hep-ph/0109068",
        "ScalePDF": "Q-scale used in evaluation of PDF's [GeV]",
        "Sigma": "vertex position (z component) error",
        "SoftDroppedP4[5]": "first entry (i = 0) is the total SoftDropped Jet 4-momenta and from i = 1 to 4 are the pruned subjets 4-momenta",
        "SoftDroppedSubJet1": "leading soft-dropped subjet",
        "SoftDroppedSubJet2": "subleading soft-dropped subjet",
        "Status": "particle status",
        "SumPT2": "sum pt^2 of tracks attached to the vertex",
        "SumPt": "isolation variable",
        "SumPtCharged": "isolation variable",
        "SumPtChargedPU": "isolation variable",
        "SumPtNeutral": "isolation variable",
        "TOuter": "time position (t component) at the edge",
        "TauTag": "0 or 1 for a particle that has been tagged as a tau",
        "Tau[5]": "N-subjettiness",
        "Trigger": "trigger word",
        "TrimmedP4[5]": "first entry (i = 0) is the total Trimmed Jet 4-momenta and from i = 1 to 4 are the trimmed subjets 4-momenta",
        "Tx": "angle of the momentum in the horizontal (x,z) plane [urad]",
        "Ty": "angle of the momentum in the verical (y,z) plane [urad]",
        "VertexIndex": "reference to vertex",
        "Weight": "weight for the event",
        "X1": 'fraction of beam momentum carried by first parton ("beam side")',
        "X2": 'fraction of beam momentum carried by second parton ("target side")',
        "XOuter": "position (x component) at the edge",
        "Xd": "X coordinate of point of closest approach to vertex",
        "YOuter": "position (y component) at the edge",
        "Yd": "Y coordinate of point of closest approach to vertex",
        "ZOuter": "position (z component) at the edge",
        "Zd": "Z coordinate of point of closest approach to vertex",
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
        def _tlorentz_vectorize(objname, form):
            # first handle RecordArray
            if {"fE", "fP"} == form.get("contents", {}).keys():
                return zip_forms(
                    {
                        "x": form["contents"]["fP"]["contents"]["fX"],
                        "y": form["contents"]["fP"]["contents"]["fY"],
                        "z": form["contents"]["fP"]["contents"]["fZ"],
                        "t": form["contents"]["fE"],
                    },
                    objname,
                    "LorentzVector",
                )
            # If there's no "content", like a NumpyArray, just return.
            # Note: this comes after checking for RecordArray.
            if "content" not in form:
                return form
            # Then recursively go through and update the form's content.
            form["content"] = _tlorentz_vectorize(objname, form["content"])
            return form

        # preprocess lorentz vectors properly (and recursively)
        for objname, form in branch_forms.items():
            branch_forms[objname] = _tlorentz_vectorize(objname, form)

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
            output[f"{name}.offsets"] = branch_forms[f"o{name}"]
            mixin = self.mixins.get(name, "NanoCollection")

            # Every delphes collection is a list
            offsets = branch_forms["o" + name]
            content = {
                k[2 * len(name) + 2 :]: branch_forms[k]
                for k in branch_forms
                if k.startswith(name + "/" + name)
            }
            output[name] = zip_forms(content, name, record_name=mixin, offsets=offsets)

            # update docstrings as needed
            # NB: must be before flattening for easier logic
            for parameter in output[name]["content"]["contents"].keys():
                if "parameters" not in output[name]["content"]["contents"][parameter]:
                    continue
                output[name]["content"]["contents"][parameter]["parameters"][
                    "__doc__"
                ] = self.docstrings.get(
                    parameter,
                    output[name]["content"]["contents"][parameter]["parameters"].get(
                        "__doc__", "no docstring available"
                    ),
                )

            # handle branches named like [4] and [5]
            output[name]["content"]["contents"] = {
                k.replace("[", "_").replace("]", ""): v
                for k, v in output[name]["content"]["contents"].items()
            }
            output[name]["content"]["parameters"].update(
                {
                    "__doc__": offsets["parameters"]["__doc__"],
                    "collection_name": name,
                }
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
