import warnings

from collections import defaultdict

from coffea.nanoevents.schemas.base import BaseSchema, zip_forms


class PHYSLITESchema(BaseSchema):
    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        zip_groups = defaultdict(list)
        for key, ak_form in branch_forms.items():
            key_fields = key.split(".")
            top_key = key_fields[0]
            sub_key = ".".join(key_fields[1:])
            objname = top_key.replace("Analysis", "").replace("AuxDyn", "")
            zip_groups[objname].append((key, sub_key))

        contents = {}
        for objname, keys in zip_groups.items():
            try:
                contents[objname] = zip_forms(
                    {sub_key: branch_forms[key] for key, sub_key in keys},
                    objname,
                )
            except NotImplementedError:
                warnings.warn(f"Can't zip collection {objname}")
        return contents
