import json
import warnings

from coffea.nanoevents.mapping.base import BaseSourceMapping, UUIDOpener
from coffea.nanoevents.util import quote, tuple_to_key


class SimplePreloadedColumnSource(dict):
    def __init__(self, columns, uuid, num_rows, object_path, **kwargs):
        self.update(columns)
        self.metadata = {"uuid": uuid, "num_rows": num_rows, "object_path": object_path}
        self.metadata.update(kwargs)


class PreloadedOpener(UUIDOpener):
    def __init__(self, uuid_pfnmap):
        super().__init__(uuid_pfnmap)

    def open_uuid(self, uuid):
        pcs = self._uuid_pfnmap[uuid]
        if str(pcs.uuid) != uuid:
            raise RuntimeError(
                f"UUID of array source {pcs} does not match expected value ({uuid})"
            )
        return pcs


class PreloadedSourceMapping(BaseSourceMapping):
    _debug = False

    def __init__(self, array_source, start, stop, cache=None, access_log=None):
        super().__init__(array_source, start, stop, cache, access_log)

    @classmethod
    def _extract_base_form(cls, column_source, force_to_i64=False):
        branch_forms = {}
        for key, branch in column_source.items():
            if "," in key or "!" in key:
                warnings.warn(
                    f"Skipping {key} because it contains characters that NanoEvents cannot accept [,!]"
                )
                continue
            form = json.loads(branch.layout.form.to_json())
            if (
                form["class"].startswith("ListOffset")
                and form["content"]["class"] == "NumpyArray"  # noqa
            ):
                form["form_key"] = quote(f"{key},!load")
                form["content"]["form_key"] = quote(f"{key},!load,!content")
                form["content"]["parameters"] = {"__doc__": key}
                if force_to_i64:
                    form["offsets"] = "i64"
            elif form["class"] == "NumpyArray":
                form["form_key"] = quote(f"{key},!load")
                form["parameters"] = {"__doc__": key}
            else:
                warnings.warn(
                    f"Skipping {key} as it is not interpretable by NanoEvents"
                )
                continue
            branch_forms[key] = form

        return {
            "class": "RecordArray",
            "fields": [key for key in branch_forms.keys()],
            "contents": [value for value in branch_forms.values()],
            "parameters": {"__doc__": "preloaded column source"},
            "form_key": "",
        }

    def key_root(self):
        return "PreloadedSourceMapping:"

    def preload_column_source(self, uuid, path_in_source, source):
        """To save a double-open when using NanoEventsFactory.from_file"""
        key = self.key_root() + tuple_to_key((uuid, path_in_source))
        self._cache[key] = source

    def get_column_handle(self, columnsource, name):
        return columnsource[name]

    def extract_column(self, columnhandle, start, stop, **kwargs):
        # make sure uproot is single-core since our calling context might not be
        return columnhandle[start:stop]

    def __len__(self):
        return self._stop - self._start

    def __iter__(self):
        raise NotImplementedError
