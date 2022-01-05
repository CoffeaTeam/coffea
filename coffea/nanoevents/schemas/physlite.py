import warnings
from collections import defaultdict
import copy
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
from coffea.nanoevents.util import quote


class PHYSLITESchema(BaseSchema):
    """PHYSLITE schema builder - work in progress.

    This is a schema for the `ATLAS DAOD_PHYSLITE derivation
    <https://gitlab.cern.ch/atlas/athena/-/blob/release/21.2.108.0/PhysicsAnalysis/DerivationFramework/DerivationFrameworkPhys/share/PHYSLITE.py>`_.
    Closely following `schemas.nanoaod.NanoAODSchema`, it is mainly build from
    naming patterns where the "Analysis" prefix has been removed, so the
    collections will be named Electrons, Muons, instead of AnalysisElectrons,
    AnalysisMunos, etc. The collection fields correspond to the "Aux" and
    "AuxDyn" columns.

    Collections are assigned mixin types according to the `mixins` mapping.
    All collections are then zipped into one `base.NanoEvents` record and returned.

    Cross references are build from ElementLink columns. Global indices are
    created dynamically, using an ``_eventindex`` field that is attached to
    each collection.
    """

    truth_collections = [
        "TruthPhotons",
        "TruthMuons",
        "TruthNeutrinos",
        "TruthTaus",
        "TruthElectrons",
        "TruthBoson",
        "TruthBottom",
        "TruthTop",
    ]
    """TRUTH3 collection names.

    TruthParticle behavior is assigned to all of them and global index forms
    for parent/children relations are created for all combinations.
    """

    mixins = {
        "Electrons": "Electron",
        "Muons": "Muon",
        "Jets": "Particle",
        "TauJets": "Particle",
        "CombinedMuonTrackParticles": "TrackParticle",
        "ExtrapolatedMuonTrackParticles": "TrackParticle",
        "GSFTrackParticles": "TrackParticle",
        "InDetTrackParticles": "TrackParticle",
        "MuonSpectrometerTrackParticles": "TrackParticle",
    }
    """Default configuration for mixin types, based on the collection name.

    The types are implemented in the `coffea.nanoevents.methods.physlite` module.
    """

    for _k in truth_collections:
        mixins[_k] = "TruthParticle"

    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        zip_groups = defaultdict(list)
        has_eventindex = defaultdict(bool)
        for key, ak_form in branch_forms.items():
            # Normal fields
            key_fields = key.split("/")[-1].split(".")
            top_key = key_fields[0]
            sub_key = ".".join(key_fields[1:])
            objname = top_key.replace("Analysis", "").replace("AuxDyn", "")

            zip_groups[objname].append(((key, sub_key), ak_form))

            # add eventindex form, based on the first single-jagged list column
            if (
                not has_eventindex[objname]
                and "List" in ak_form["class"]
                and "List" not in ak_form["content"]["class"]
            ):
                zip_groups[objname].append(
                    ((key, "_eventindex"), self._create_eventindex_form(ak_form, key))
                )
                has_eventindex[objname] = True

        # zip the forms
        contents = {}
        for objname, keys_and_form in zip_groups.items():
            try:
                contents[objname] = zip_forms(
                    {sub_key: form for (key, sub_key), form in keys_and_form},
                    objname,
                    self.mixins.get(objname, None),
                    bypass=True,
                )
                content = contents[objname]["content"]
                content["parameters"] = dict(
                    content.get("parameters", {}), collection_name=objname
                )
            except NotImplementedError:
                warnings.warn(f"Can't zip collection {objname}")
        return contents

    @staticmethod
    def _create_eventindex_form(base_form, key):
        form = copy.deepcopy(base_form)
        form["content"] = {
            "class": "NumpyArray",
            "parameters": {},
            "form_key": quote(f"{key},!load,!eventindex,!content"),
            "itemsize": 8,
            "primitive": "int64",
        }
        return form

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import physlite

        return physlite.behavior
