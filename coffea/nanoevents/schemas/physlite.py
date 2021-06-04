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

    Cross references are build from ElementLink columns where the possible
    index -> target collection relations are currently hard coded in the
    `cross_reference_indices` and `cross_reference_elementlinks` mappings.
    """

    _hack_for_elementlink_int64 = True

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

    cross_reference_indices = {
        ("Muons", "combinedTrackParticleLink.m_persIndex"): [
            "CombinedMuonTrackParticles"
        ]
    }
    """Possible relations for single-jagged ElementLinks"""

    cross_reference_elementlinks = {
        ("Electrons", "trackParticleLinks"): ["GSFTrackParticles"],
        ("HLT_e7_lhmedium_nod0_mu24", "TrigMatchedObjects"): ["Electrons", "Muons"],
    }
    """Possible relations for double-jagged ElementLinks"""
    for _k in truth_collections:
        cross_reference_elementlinks[(_k, "childLinks")] = truth_collections
        cross_reference_elementlinks[(_k, "parentLinks")] = truth_collections

    link_load_columns = {
        "CombinedMuonTrackParticles": "CombinedMuonTrackParticlesAuxDyn.z0",
        "GSFTrackParticles": "GSFTrackParticlesAuxDyn.z0",
        "Electrons": "AnalysisElectronsAuxDyn.pt",
        "Muons": "AnalysisMuonsAuxDyn.pt",
    }
    """Columns to load for extracting offsets when creating global indices"""
    for _k in truth_collections:
        link_load_columns[_k] = f"{_k}AuxDyn.px"

    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        zip_groups = defaultdict(list)
        for key, ak_form in branch_forms.items():
            # Normal fields
            key_fields = key.split("/")[-1].split(".")
            top_key = key_fields[0]
            sub_key = ".".join(key_fields[1:])
            objname = top_key.replace("Analysis", "").replace("AuxDyn", "")

            # temporary hack to have the correct type for the ElementLinks
            # (uproot loses the type information somewhere on the way and they end up int64)
            if self._hack_for_elementlink_int64:
                try:
                    for k in ["m_persIndex", "m_persKey"]:
                        form = ak_form["content"]["content"]["contents"][k]
                        form["itemsize"] = 8
                        form["primitive"] = "int64"
                except KeyError:
                    pass

            zip_groups[objname].append(((key, sub_key), ak_form))

            # Global indices
            for cross_references in [
                self.cross_reference_indices,
                self.cross_reference_elementlinks,
            ]:
                if (objname, sub_key) in cross_references:
                    for linkto_collection in cross_references[(objname, sub_key)]:
                        linkto_key = self.link_load_columns[linkto_collection]
                        if cross_references is self.cross_reference_indices:
                            form = self._create_global_index_form(
                                ak_form, key, linkto_key
                            )
                        elif cross_references is self.cross_reference_elementlinks:
                            form = self._create_global_index_form_elementlink(
                                ak_form, key, linkto_key
                            )
                        zip_groups[objname].append(
                            ((key, sub_key + "__G__" + linkto_collection), form)
                        )

        contents = {}
        for objname, keys_and_form in zip_groups.items():
            try:
                contents[objname] = zip_forms(
                    {sub_key: form for (key, sub_key), form in keys_and_form},
                    objname,
                    self.mixins.get(objname, None),
                )
                content = contents[objname]["content"]
                content["parameters"] = dict(
                    content.get("parameters", {}), collection_name=objname
                )
            except NotImplementedError:
                warnings.warn(f"Can't zip collection {objname}")
        return contents

    @staticmethod
    def _create_global_index_form(base_form, key, linkto_key):
        form = copy.deepcopy(base_form)
        form["content"]["form_key"] = quote(
            f"{key},!load,{linkto_key},!load,!offsets,!local2global"
        )
        form["content"]["itemsize"] = 8
        form["content"]["primitive"] = "int64"
        return form

    @staticmethod
    def _create_global_index_form_elementlink(base_form, key, linkto_key):
        form = copy.deepcopy(base_form)
        record = form["content"]["content"]["contents"]
        form["content"]["content"] = record["m_persIndex"]
        form["content"]["content"]["form_key"] = quote(
            f"{key},!load,m_persIndex,!item,{linkto_key},!load,!offsets,!local2global"
        )
        form["content"]["form_key"] = quote(
            # the !skip,{linkto_key} is a hack to avoid having the same key as the actual elementlink
            f"{key},!load,!content,!skip,{linkto_key}"
        )
        form["content"]["content"]["itemsize"] = 8
        form["content"]["content"]["primitive"] = "int64"
        return form

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import physlite

        return physlite.behavior
