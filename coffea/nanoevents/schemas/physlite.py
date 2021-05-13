import warnings
from collections import defaultdict
import copy
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
from coffea.nanoevents.util import quote


class PHYSLITESchema(BaseSchema):

    mixins = {
        "Electrons": "xAODParticle",
        "Muons": "xAODParticle",
        "Jets": "xAODParticle",
        "TauJets": "xAODParticle",
        "CombinedMuonTrackParticles": "xAODTrackParticle",
        "ExtrapolatedMuonTrackParticles": "xAODTrackParticle",
        "GSFTrackParticles": "xAODTrackParticle",
        "InDetTrackParticles": "xAODTrackParticle",
        "MuonSpectrometerTrackParticles": "xAODTrackParticle",
    }

    # create global indices for single-jagged arrays after cross referencing
    # for the target collection an arbitrary column (e.g z0) has to be chosen to extract the offsets
    cross_reference_indices = {
        (
            "Muons",
            "combinedTrackParticleLink.m_persIndex",
        ): "CombinedMuonTrackParticlesAuxDyn.z0",
    }
    # create global indices for double-jagged arrays after cross referencing
    # here we will resolve ".m_persIndex" since these branches are not split
    cross_reference_elementlinks = {
        ("Electrons", "trackParticleLinks"): "GSFTrackParticlesAuxDyn.z0",
    }

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
            zip_groups[objname].append(((key, sub_key), ak_form))

            # Global indices
            for cross_references in [
                self.cross_reference_indices,
                self.cross_reference_elementlinks,
            ]:
                if (objname, sub_key) in cross_references:
                    linkto_key = cross_references[(objname, sub_key)]
                    if cross_references is self.cross_reference_indices:
                        form = self._create_global_index_form(ak_form, key, linkto_key)
                    elif cross_references is self.cross_reference_elementlinks:
                        form = self._create_global_index_form_elementlink(
                            ak_form, key, linkto_key
                        )
                    zip_groups[objname].append(((key, sub_key + "G"), form))

        contents = {}
        for objname, keys_and_form in zip_groups.items():
            try:
                contents[objname] = zip_forms(
                    {sub_key: form for (key, sub_key), form in keys_and_form},
                    objname,
                    self.mixins.get(objname, None),
                )
            except NotImplementedError:
                warnings.warn(f"Can't zip collection {objname}")
        return contents

    @staticmethod
    def _create_global_index_form(base_form, key, linkto_key):
        form = copy.deepcopy(base_form)
        form["content"]["form_key"] = quote(
            f"{key},!load,{linkto_key},!load,!offsets,!to_numpy,!local2global"
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
            f"{key},!load,m_persIndex,!item,{linkto_key},!load,!offsets,!to_numpy,!local2global"
        )
        form["content"]["form_key"] = quote(
            # the !data is a hack to avoid having the same key as the actual elementlink
            f"{key},!load,!content,!data"
        )
        form["content"]["content"]["itemsize"] = 8
        form["content"]["content"]["primitive"] = "int64"
        return form

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import physlite

        return physlite.behavior
