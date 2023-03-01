import warnings
import uproot
import awkward
import json
from coffea.nanoevents.mapping.base import UUIDOpener, BaseSourceMapping
from coffea.nanoevents.util import quote, tuple_to_key


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


class CannotBeNanoEvents(Exception):
    pass


def _lazify_form(form, prefix, docstr=None):
    if not isinstance(form, dict) or "class" not in form:
        raise RuntimeError("form should have been normalized by now")

    parameters = _lazify_parameters(form.get("parameters", {}), docstr=docstr)
    if form["class"].startswith("ListOffset"):
        # awkward will add !offsets
        form["form_key"] = quote(prefix)
        form["content"] = _lazify_form(
            form["content"], prefix + ",!content", docstr=docstr
        )
    elif form["class"] == "NumpyArray":
        form["form_key"] = quote(prefix)
        if parameters:
            form["parameters"] = parameters
    elif form["class"] == "RegularArray":
        form["content"] = _lazify_form(
            form["content"], prefix + ",!content", docstr=docstr
        )
        if parameters:
            form["parameters"] = parameters
    elif form["class"] == "RecordArray":
        for field in list(form["contents"]):
            if "," in field or "!" in field:
                raise CannotBeNanoEvents(
                    f"A subform contains a field with invalid characters: {field}"
                )
            print(form["contents"])
            print(form["fields"])
            form["contents"][field] = _lazify_form(
                form["contents"][field], prefix + f",{field},!item"
            )
        if parameters:
            form["parameters"] = parameters
    else:
        raise CannotBeNanoEvents("Unknown form")
    return form


def _lazify_parameters(form_parameters, docstr=None):
    parameters = {}
    if "__array__" in form_parameters:
        parameters["__array__"] = form_parameters["__array__"]
    if docstr is not None:
        parameters["__doc__"] = docstr
    return parameters


class UprootSourceMapping(BaseSourceMapping):
    _debug = True
    _fix_awkward_form_of_iter = False

    def __init__(self, fileopener, cache=None, access_log=None):
        super(UprootSourceMapping, self).__init__(fileopener, cache, access_log)

    @classmethod
    def _extract_base_form(cls, tree, iteritems_options={}):
        branch_forms = {}
        for key, branch in tree.iteritems(**iteritems_options):
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
                # The branch is split and its sub-branches will be enumerated by tree.iteritems
                continue
            if isinstance(
                branch.interpretation,
                uproot.interpretation.identify.UnknownInterpretation,
            ):
                warnings.warn(f"Skipping {key} as it is not interpretable by Uproot")
                continue
            try:
                form = branch.interpretation.awkward_form(None)
            except uproot.interpretation.objects.CannotBeAwkward:
                warnings.warn(
                    f"Skipping {key} as it is it cannot be represented as an Awkward array"
                )
                continue
            # until awkward-forth is available, this fixer is necessary
            if cls._fix_awkward_form_of_iter:
                form = uproot._util.recursively_fix_awkward_form_of_iter(
                    awkward, branch.interpretation, form
                )
            form = json.loads(
                form.to_json()
            )  # normalizes form (expand NumpyArray classes)
            try:
                form = _lazify_form(form, f"{key},!load", docstr=branch.title)
            except CannotBeNanoEvents as ex:
                warnings.warn(
                    f"Skipping {key} as it is not interpretable by NanoEvents\nDetails: {ex}"
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
