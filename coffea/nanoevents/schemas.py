from coffea.nanoevents import transforms
from coffea.nanoevents.util import quote, concat


def listarray_form(content, offsets):
    if offsets["class"] != "NumpyArray":
        raise ValueError
    if offsets["primitive"] == "int32":
        arrayclass = "ListOffsetArray32"
        offsetstype = "i32"
    elif offsets["primitive"] == "int64":
        arrayclass = "ListOffsetArray64"
        offsetstype = "i64"
    else:
        raise ValueError("Unrecognized offsets data type")
    return {
        "class": arrayclass,
        "offsets": offsetstype,
        "content": content,
        "form_key": concat(offsets["form_key"], "!skip"),
    }


def zip_forms(forms, name, record_name=None, offsets=None):
    if not isinstance(forms, dict):
        raise ValueError("Expected a dictionary")
    if all(form["class"].startswith("ListOffsetArray") for form in forms.values()):
        first = next(iter(forms.values()))
        if not all(form["class"] == first["class"] for form in forms.values()):
            raise ValueError
        if not all(form["offsets"] == first["offsets"] for form in forms.values()):
            raise ValueError
        record = {
            "class": "RecordArray",
            "contents": {k: form["content"] for k, form in forms.items()},
            "form_key": quote("!invalid," + name),
        }
        if record_name is not None:
            record["parameters"] = {"__record__": record_name}
        if offsets is None:
            return {
                "class": first["class"],
                "offsets": first["offsets"],
                "content": record,
                "form_key": first["form_key"],
            }
        else:
            return listarray_form(record, offsets)
    elif all(form["class"] == "NumpyArray" for form in forms.values()):
        record = {
            "class": "RecordArray",
            "contents": {k: form for k, form in forms.items()},
            "form_key": quote("!invalid," + name),
        }
        if record_name is not None:
            record["parameters"] = {"__record__": record_name}
        return record
    else:
        raise NotImplementedError("Cannot zip forms")


def nest_jagged_forms(parent, child, counts_name, name):
    """Place child listarray inside parent listarray as a double-jagged array"""
    if not parent["class"].startswith("ListOffsetArray"):
        raise ValueError
    if parent["content"]["class"] != "RecordArray":
        raise ValueError
    if not child["class"].startswith("ListOffsetArray"):
        raise ValueError
    counts = parent["content"]["contents"][counts_name]
    offsets = transforms.counts2offsets_form(counts)
    inner = listarray_form(child["content"], offsets)
    parent["content"]["contents"][name] = inner


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
            "form_key": "",
        }

    @property
    def form(self):
        """Awkward1 form of this schema"""
        return self._form

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import base

        return base.behavior


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
                    k[len(name) + 1 :]: branch_forms[k]
                    for k in branch_forms
                    if k.startswith(name + "_")
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
            elif "o" + name in branch_forms:
                # list singleton, can use branch's own offsets
                output[name] = branch_forms[name]
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
                output[name]["parameters"].update({"collection_name": name})

        return output

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import nanoaod

        return nanoaod.behavior


class TreeMakerSchema(BaseSchema):
    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        # Turn any special classes into the appropriate awkward form
        composite_objects = list(set(k.split("/")[0] for k in branch_forms if "/" in k))
        for objname in composite_objects:
            # grab the * from "objname/objname.*"
            components = set(
                k[2 * len(objname) + 2 :]
                for k in branch_forms
                if k.startswith(objname + "/")
            )
            if components == {
                "fCoordinates.fPt",
                "fCoordinates.fEta",
                "fCoordinates.fPhi",
                "fCoordinates.fE",
            }:
                form = zip_forms(
                    {
                        "pt": branch_forms.pop(f"{objname}/{objname}.fCoordinates.fPt"),
                        "eta": branch_forms.pop(
                            f"{objname}/{objname}.fCoordinates.fEta"
                        ),
                        "phi": branch_forms.pop(
                            f"{objname}/{objname}.fCoordinates.fPhi"
                        ),
                        "energy": branch_forms.pop(
                            f"{objname}/{objname}.fCoordinates.fE"
                        ),
                    },
                    objname,
                    "PtEtaPhiELorentzVector",
                )
                print("PtEtaPhiELorentzVector:", objname)
                branch_forms[objname] = form
            elif components == {
                "fCoordinates.fX",
                "fCoordinates.fY",
                "fCoordinates.fZ",
            }:
                form = zip_forms(
                    {
                        "x": branch_forms.pop(f"{objname}/{objname}.fCoordinates.fX"),
                        "y": branch_forms.pop(f"{objname}/{objname}.fCoordinates.fY"),
                        "z": branch_forms.pop(f"{objname}/{objname}.fCoordinates.fZ"),
                    },
                    objname,
                    "Point",
                )
                print("Point:", objname)
                branch_forms[objname] = form
            else:
                raise ValueError(
                    f"Unrecognized class with split branches: {components}"
                )

        collections = [
            "Electrons",
            "GenElectrons",
            "GenJets",
            "GenJetsAK8",
            "GenMuons",
            "GenParticles",
            "GenTaus",
            "Jets",
            "JetsAK8",
            "JetsAK8_subjets",
            "JetsAK15",
            "Muons",
            "PrimaryVertices",
            "TAPElectronTracks",
            "TAPMuonTracks",
            "TAPPionTracks",
            "Tracks",
        ]
        for cname in collections:
            items = sorted(k for k in branch_forms if k.startswith(cname + "_"))
            if len(items) == 0:
                continue
            if cname == "JetsAK8":
                items = [k for k in items if not k.startswith("JetsAK8_subjets")]
                items.append("JetsAK8_subjetsOffsets")  # FIXME: actually counts
            if cname == "JetsAK8_subjets":
                items = [k for k in items if not k.endswith("Offsets")]
            if cname not in branch_forms:
                collection = zip_forms(
                    {k[len(cname) + 1]: branch_forms.pop(k) for k in items}, cname
                )
                branch_forms[cname] = collection
            else:
                collection = branch_forms[cname]
                if not collection["class"].startswith("ListOffsetArray"):
                    raise NotImplementedError(
                        f"{cname} isn't a jagged array, not sure what to do"
                    )
                for item in items:
                    itemname = item[len(cname) + 1 :]
                    collection["content"]["contents"][itemname] = branch_forms.pop(
                        item
                    )["content"]

        nest_jagged_forms(
            branch_forms["JetsAK8"],
            branch_forms.pop("JetsAK8_subjets"),
            "subjetsOffsets",
            "subjets",
        )
        return branch_forms

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import base, vector

        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        return behavior
