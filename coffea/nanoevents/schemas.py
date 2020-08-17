from coffea.nanoevents import transforms
from coffea.nanoevents.util import quote, concat


class BaseSchema:
    """Base schema builder

    The basic schema is essentially unchanged from the original ROOT file.
    A top-level `base.NanoEvents` object is returned, where each original branch
    form is accessible as a direct descendant.
    """
    behavior = {}

    def __init__(self, base_form):
        params = dict(base_form.get("parameters", {}))
        params["__record__"] = "NanoEvents"
        params.setdefault("metadata", {})
        self._form = {
            "class": "RecordArray",
            "contents": base_form["contents"],
            "parameters": params,
            "form_key": quote("!invalid"),
        }

    @property
    def form(self):
        """Awkward1 form of this schema"""
        return self._form

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods.base import behavior

        return behavior


class NanoAODSchema(BaseSchema):
    """NanoAOD schema builder

    The NanoAOD schema is built from all branches found in the supplied file, based on
    the naming pattern of the branches. The following additional arrays are constructed:

    - Any branches named ``n{name}`` are assumed to be counts branches and converted to offsets ``o{name}``
    - Any local index branches with names matching ``{source}_{target}Idx*`` are converted to global indexes for the event chunk (postfix ``G``)
    - Any `nested_items` are constructed, if the necessary branches are available
    - Any `special_items` are constructed, if the necessary branches are available

    From those arrays, NanoAOD collections are formed as collections of branches grouped by name, where:

    - one branch exists named ``name`` and no branches start with ``name_``, interpreted as a single flat array;
    - one branch exists named ``name``, one named ``n{name}``, and no branches start with ``name_``, interpreted as a single jagged array;
    - no branch exists named ``{name}`` and many branches start with ``name_*``, interpreted as a flat table; or
    - one branch exists named ``n{name}`` and many branches start with ``name_*``, interpreted as a jagged table.

    Collections are assigned mixin types according to the `mixins` mapping.
    All collections are then zipped into one `base.NanoEvents` record and returned.
    """

    mixins = {
        "CaloMET": "MissingET",
        "ChsMET": "MissingET",
        "GenMET": "MissingET",
        "MET": "MissingET",
        "METFixEE2017": "MissingET",
        "PuppiMET": "MissingET",
        "RawMET": "MissingET",
        "TkMET": "MissingET",
        # pseudo-lorentz: pt, eta, phi, mass=0
        "IsoTrack": "PtEtaPhiMCollection",
        "SoftActivityJet": "PtEtaPhiMCollection",
        "TrigObj": "PtEtaPhiMCollection",
        # True lorentz: pt, eta, phi, mass
        "FatJet": "FatJet",
        "GenDressedLepton": "PtEtaPhiMCollection",
        "GenJet": "PtEtaPhiMCollection",
        "GenJetAK8": "FatJet",
        "Jet": "Jet",
        "LHEPart": "PtEtaPhiMCollection",
        "SV": "PtEtaPhiMCollection",
        "SubGenJetAK8": "PtEtaPhiMCollection",
        "SubJet": "PtEtaPhiMCollection",
        # Candidate: lorentz + charge
        "Electron": "Electron",
        "Muon": "Muon",
        "Photon": "Photon",
        "FsrPhoton": "FsrPhoton",
        "Tau": "Tau",
        "GenVisTau": "GenVisTau",
        # special
        "GenPart": "GenParticle",
    }
    """Default configuration for mixin types, based on the collection name.

    The types are implemented in the `coffea.nanoevents.methods.nanoaod` module.
    """
    nested_items = {
        "FatJet_subJetIdxG": ["FatJet_subJetIdx1G", "FatJet_subJetIdx2G"],
        "Jet_muonIdxG": ["Jet_muonIdx1G", "Jet_muonIdx2G"],
        "Jet_electronIdxG": ["Jet_electronIdx1G", "Jet_electronIdx2G"],
    }
    """Default nested collections, where nesting is accomplished by a fixed-length set of indexers"""
    special_items = {
        "GenPart_distinctParentIdxG": (
            transforms.distinctParent_form,
            ("GenPart_genPartIdxMotherG", "GenPart_pdgId"),
        ),
        "GenPart_childrenIdxG": (
            transforms.children_form,
            ("oGenPart", "GenPart_genPartIdxMotherG",),
        ),
        "GenPart_distinctChildrenIdxG": (
            transforms.children_form,
            ("oGenPart", "GenPart_distinctParentIdxG",),
        ),
    }
    """Default special arrays, where the callable and input arrays are specified in the value"""

    def __init__(self, base_form, version="6"):
        self._version = version
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])
        self._form["parameters"]["metadata"]["version"] = self._version

    def _build_collections(self, branch_forms):
        # parse into high-level records (collections, list collections, and singletons)
        collections = set(k.split("_")[0] for k in branch_forms)
        collections -= set(
            k for k in collections if k.startswith("n") and k[1:] in collections
        )

        # Create offsets virtual arrays
        for name in collections:
            if "n" + name in branch_forms:
                branch_forms["o" + name] = transforms.counts2offsets_form(
                    branch_forms["n" + name]
                )

        # Create global index virtual arrays for indirection
        for name in collections:
            indexers = filter(lambda k: k.startswith(name) and "Idx" in k, branch_forms)
            for k in list(indexers):
                target = k[len(name) + 1 : k.find("Idx")]
                target = target[0].upper() + target[1:]
                if target not in collections:
                    raise RuntimeError(
                        "Parsing indexer %s, expected to find collection %s but did not"
                        % (k, target)
                    )
                branch_forms[k + "G"] = transforms.local2global_form(
                    branch_forms[k], branch_forms["o" + target]
                )

        # Create nested indexer from Idx1, Idx2, ... arrays
        for name, indexers in self.nested_items.items():
            if all(idx in branch_forms for idx in indexers):
                branch_forms[name] = transforms.nestedindex_form(
                    [branch_forms[idx] for idx in indexers]
                )

        # Create any special arrays
        for name, (fcn, args) in self.special_items.items():
            if all(k in branch_forms for k in args):
                branch_forms[name] = fcn(*(branch_forms[k] for k in args))

        output = {}
        for name in collections:
            mixin = self.mixins.get(name, "NanoCollection")
            if "o" + name in branch_forms and name not in branch_forms:
                # list collection
                offsets = branch_forms["o" + name]
                content = {
                    k[len(name) + 1 :]: branch_forms[k]["content"]
                    for k in branch_forms
                    if k.startswith(name + "_")
                }
                output[name] = {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": {
                        "class": "RecordArray",
                        "contents": content,
                        "parameters": {
                            "__doc__": offsets["parameters"]["__doc__"],
                            "__record__": mixin,
                            "collection_name": name,
                        },
                        "form_key": quote("!invalid"),
                    },
                    "form_key": concat(offsets["form_key"], "!skip"),
                }
            elif "o" + name in branch_forms:
                # list singleton
                offsets = branch_forms["o" + name]
                content = branch_forms[name]["content"]
                output[name] = {
                    "class": "ListOffsetArray64",
                    "offsets": "i64",
                    "content": content,
                    "parameters": {
                        # This makes more sense as offsets doc but it seems that is empty
                        "__doc__": content["parameters"]["__doc__"],
                        "__array__": mixin,
                        "collection_name": name,
                    },
                    "form_key": concat(offsets["form_key"], "!skip"),
                }
            elif name in branch_forms:
                # singleton
                output[name] = branch_forms[name]
            else:
                # simple collection
                content = {
                    k[len(name) + 1 :]: branch_forms[k]
                    for k in branch_forms
                    if k.startswith(name + "_")
                }
                output[name] = {
                    "class": "RecordArray",
                    "contents": content,
                    "parameters": {"__record__": mixin, "collection_name": name},
                    "form_key": quote("!invalid"),
                }

        return output

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods.nanoaod import behavior

        return behavior
