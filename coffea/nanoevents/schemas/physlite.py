import warnings
from collections import defaultdict
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
    cross_reference_indices = {
        (
            "Muons",
            "combinedTrackParticleLink.m_persIndex",
        ): "CombinedMuonTrackParticlesAuxDyn.z0",
    }
    # create global indices for double-jagged arrays after cross referencing
    # here we need to resolve ".m_persIndex" since these branches are not split
    cross_reference_elementlinks = {
        ("Electrons", "trackParticleLinks"): "GSFTrackParticlesAuxDyn.z0",
    }

    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        zip_groups = defaultdict(list)
        for key, ak_form in branch_forms.items():
            key_fields = key.split("/")[-1].split(".")
            top_key = key_fields[0]
            sub_key = ".".join(key_fields[1:])
            objname = top_key.replace("Analysis", "").replace("AuxDyn", "")
            zip_groups[objname].append(((key, sub_key), ak_form))
            for cross_references in [
                self.cross_reference_indices,
                self.cross_reference_elementlinks,
            ]:
                if (objname, sub_key) in cross_references:
                    linkto_key = cross_references[(objname, sub_key)]
                    form = dict(ak_form)
                    if cross_references is self.cross_reference_indices:
                        form["content"]["form_key"] = quote(
                            f"{key},!load,{linkto_key},!load,!offsets,!to_numpy,!local2global"
                        )
                    elif cross_references is self.cross_reference_elementlinks:
                        form["content"]["content"] = form["content"]["content"][
                            "contents"
                        ]["m_persIndex"]
                        form["content"]["content"]["form_key"] = quote(
                            f"{key},!load,m_persIndex,!item,{linkto_key},!load,!offsets,!to_numpy,!local2global"
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

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import physlite

        return physlite.behavior
