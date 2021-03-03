import warnings
from cachetools import LRUCache
from collections.abc import Mapping
import uproot
import awkward
import numpy
import json
from coffea.nanoevents.mapping.base import UUIDOpener, BaseSourceMapping
from coffea.nanoevents.util import quote, key_to_tuple, tuple_to_key


class TrivialUprootOpener(UUIDOpener):
    def __init__(self, uuid_pfnmap, uproot_options={}):
        super(TrivialUprootOpener, self).__init__(uuid_pfnmap)
        self._uproot_options = uproot_options

    def open_uuid(self, uuid):
        pfn = self._uuid_pfnmap[uuid]
        rootdir = uproot.open(pfn, **self._uproot_options)
        if str(rootdir.file.uuid) != uuid:
            raise RuntimeError(
                f"UUID of file {pfn} does not match expected value ({uuid})"
            )
        return rootdir


class UprootSourceMapping(BaseSourceMapping):
    _debug = False

    def __init__(self, fileopener, cache=None, access_log=None):
        super(UprootSourceMapping, self).__init__(fileopener, cache, access_log)

    @classmethod
    def _extract_base_form(cls, tree):
        branch_forms = {}
        for key, branch in tree.iteritems():
            if key in branch_forms:
                warnings.warn(
                    f"Found duplicate branch {key} in {tree}, taking first instance"
                )
                continue
            if "," in key or "!" in key:
                warnings.warn(
                    f"Skipping {key} because it contains characters that NanoEvents cannot accept [,!]"
                )
                continue
            if len(branch):
                continue
            form = branch.interpretation.awkward_form(None)
            form = uproot._util.awkward_form_remove_uproot(awkward, form)
            form = json.loads(form.tojson())
            if (
                form["class"].startswith("ListOffset")
                and form["content"]["class"] == "NumpyArray"  # noqa
            ):
                form["form_key"] = quote(f"{key},!load")
                form["content"]["form_key"] = quote(f"{key},!load,!content")
                form["content"]["parameters"] = {"__doc__": branch.title}
            elif form["class"] == "NumpyArray":
                form["form_key"] = quote(f"{key},!load")
                form["parameters"] = {"__doc__": branch.title}
            else:
                warnings.warn(
                    f"Skipping {key} as it is not interpretable by NanoEvents"
                )
                continue
            branch_forms[key] = form

        return {
            "class": "RecordArray",
            "contents": branch_forms,
            "parameters": {"__doc__": tree.title},
            "form_key": "",
        }

    def key_root(self):
        return "UprootSourceMapping:"

    def preload_column_source(self, uuid, path_in_source, source):
        """To save a double-open when using NanoEventsFactory.from_file"""
        key = self.key_root() + tuple_to_key((uuid, path_in_source))
        self._cache[key] = source

    def get_column_handle(self, columnsource, name):
        return columnsource[name]

    def extract_column(self, columnhandle, start, stop):
        # make sure uproot is single-core since our calling context might not be
        return columnhandle.array(
            entry_start=start,
            entry_stop=stop,
            decompression_executor=uproot.source.futures.TrivialExecutor(),
            interpretation_executor=uproot.source.futures.TrivialExecutor(),
        )

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
