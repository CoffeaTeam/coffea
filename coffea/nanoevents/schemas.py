import copy


class BaseSchema:
    def __init__(self):
        pass

    def __call__(self, base_form, partition_key):
        return base_form


class NanoAODSchema(BaseSchema):
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
    """Default configuration for mixin types, based on the collection name."""
    nested_items = {
        "FatJet_subJetIdxG": ["FatJet_subJetIdx1G", "FatJet_subJetIdx2G"],
        "Jet_muonIdxG": ["Jet_muonIdx1G", "Jet_muonIdx2G"],
        "Jet_electronIdxG": ["Jet_electronIdx1G", "Jet_electronIdx2G"],
    }
    """Default nested collections, where nesting is accomplished by a fixed-length set of indexers"""
    special_items = {
        "GenPart_distinctParentIdxG": (
            "!distinctParent",
            ("GenPart_genPartIdxMotherG", "GenPart_pdgId"),
        ),
        "GenPart_childrenIdxG": (
            "!children",
            ("oGenPart", "GenPart_genPartIdxMotherG"),
        ),
        "GenPart_distinctChildrenIdxG": (
            "!children",
            ("oGenPart", "GenPart_distinctParentIdxG"),
        ),
    }
    """Default special arrays, where the callable and input arrays are specified in the value"""

    def __init__(self):
        """Build NanoAOD form

        The NanoAOD form is built from all branches found in the supplied file, based on
        the naming pattern of the branches. The following additional arrays are constructed:

        - Any branches named ``n{name}`` are assumed to be counts branches and converted to offsets ``o{name}``
        - Any local index branches with names matching ``{source}_{target}Idx*`` are converted to global indexes for the event chunk
        - Any `NanoEventsFactory.nested_items` are constructed, if the necessary branches are available
        - Any `NanoEventsFactory.special_items` are constructed, if the necessary branches are available

        From those arrays, NanoAOD collections are formed as collections of branches grouped by name, where:

        - one branch exists named ``name`` and no branches start with ``name_``, interpreted as a single flat array;
        - one branch exists named ``name``, one named ``n{name}``, and no branches start with ``name_``, interpreted as a single jagged array;
        - no branch exists named ``{name}`` and many branches start with ``name_*``, interpreted as a flat table; or
        - one branch exists named ``n{name}`` and many branches start with ``name_*``, interpreted as a jagged table.

        All collections are then zipped into one `NanoEvents` record and returned.
        """
        pass

    def _build_collections(self, arrays, partiton_key):
        import awkward1

        # parse into high-level records (collections, list collections, and singletons)
        collections = set(k.split("_")[0] for k in arrays)
        collections -= set(
            k for k in collections if k.startswith("n") and k[1:] in collections
        )

        # Create offsets virtual arrays
        for name in collections:
            if "n" + name in arrays:
                array = copy.copy(arrays["n" + name])
                array["form_key"] += ",!counts2offsets"
                arrays["o" + name] = array

        # Create global index virtual arrays for indirection
        for name in collections:
            indexers = filter(lambda k: k.startswith(name) and "Idx" in k, arrays)
            for k in list(indexers):
                target = k[len(name) + 1 : k.find("Idx")]
                target = target[0].upper() + target[1:]
                if target not in collections:
                    raise RuntimeError(
                        "Parsing indexer %s, expected to find collection %s but did not"
                        % (k, target)
                    )
                arrays[k + "G"] = {
                    "form_key": ",".join(
                        [
                            arrays[k]["form_key"],
                            arrays["o" + target]["form_key"],
                            "!local2global",
                        ]
                    ),
                }

        # Create nested indexer from Idx1, Idx2, ... arrays
        for name, indexers in self.nested_items.items():
            if all(idx in arrays for idx in indexers):
                arrays[name] = self._nestedindex(
                    [arrays[idx] for idx in indexers], name
                )

        # Create any special arrays
        for name, (fcn, args) in self.special_items.items():
            if all(k in arrays for k in args):
                generator = fcn(*(arrays[k] for k in args))
                arrays[name] = awkward1.layout.VirtualArray(
                    generator,
                    self._cache,
                    cache_key="/".join([self._partition_key, fcn.__name__, name]),
                )

        def collectionfactory(name):
            mixin = self._mixin_map.get(name, "NanoCollection")
            if "o" + name in arrays and name not in arrays:
                # list collection
                offsets = arrays["o" + name]
                content = {
                    k[len(name) + 1 :]: arrays[k]
                    for k in arrays
                    if k.startswith(name + "_")
                }
                recordparams = {
                    "__doc__": offsets.parameters["__doc__"],
                    "__record__": mixin,
                    "events_key": self._partition_key,
                    "collection_name": name,
                }
                form = awkward1.forms.ListOffsetForm(
                    "i32",
                    awkward1.forms.RecordForm(
                        {k: v.form for k, v in content.items()}, parameters=recordparams
                    ),
                )
                generator = awkward1.layout.ArrayGenerator(
                    self._listarray,
                    (offsets, content, recordparams),
                    {},
                    form=form,
                    length=len(self),
                )
                source = "runtime"
                return awkward1.layout.VirtualArray(
                    generator,
                    self._cache,
                    cache_key="/".join([self._partition_key, source, name]),
                    parameters=recordparams,
                )
            elif "o" + name in arrays:
                # list singleton
                offsets = arrays["o" + name]
                content = arrays[name]
                params = {
                    # This makes more sense as offsets doc but it seems that is empty
                    "__doc__": content.parameters["__doc__"],
                    "__array__": mixin,
                    "events_key": self._partition_key,
                    "collection_name": name,
                }
                form = awkward1.forms.ListOffsetForm(
                    "i32", content.form, parameters=params
                )
                generator = awkward1.layout.ArrayGenerator(
                    self._listarray,
                    (offsets, content, params),
                    {},
                    form=form,
                    length=len(self),
                )
                source = "runtime"
                return awkward1.layout.VirtualArray(
                    generator,
                    self._cache,
                    cache_key="/".join([self._partition_key, source, name]),
                    parameters=params,
                )
            elif name in arrays:
                # singleton
                return arrays[name]
            else:
                # simple collection
                content = {
                    k[len(name) + 1 :]: arrays[k]
                    for k in arrays
                    if k.startswith(name + "_")
                }
                return awkward1.layout.RecordArray(
                    content,
                    parameters={
                        "__record__": mixin,
                        "events_key": self._partition_key,
                        "collection_name": name,
                    },
                )

        return {name: collectionfactory(name) for name in collections}

    def __call__(self, base_form, partition_key):
        params = {
            "__record__": "NanoEvents",
            "partition_key": partition_key,
        }
        params.update(base_form["parameters"])
        return {
            "class": "RecordArray",
            "contents": self._build_collections(base_form["contents"], partition_key),
            "parameters": params,
        }
