import warnings
import json
import awkward1
import uproot4
from coffea.nanoevents.util import quote, key_to_tuple, tuple_to_key
from coffea.nanoevents.mapping import UprootSourceMapping, CachedMapping
from coffea.nanoevents.schemas import BaseSchema, NanoAODSchema


class NanoEventsFactory:
    """A factory class to build NanoEvents objects

    """

    def __init__(self, schema, mapping, partition_key, cache=None):
        self._schema = schema
        self._mapping = mapping
        self._partition_key = partition_key
        self._cache = cache
        self._events = None

    def __getstate__(self):
        return {
            "schema": self._schema,
            "mapping": self._mapping,
            "partition_key": self._partition_key,
        }

    def __setstate__(self, state):
        self._schema = state["schema"]
        self._mapping = state["mapping"]
        self._partition_key = state["partition_key"]
        self._cache = None
        self._events = None

    @classmethod
    def from_file(
        cls,
        file,
        treepath="/Events",
        entry_start=None,
        entry_stop=None,
        runtime_cache=None,
        persistent_cache=None,
        schemaclass=NanoAODSchema,
        metadata=None,
    ):
        """Quickly build NanoEvents from a file

        Parameters
        ----------
            file : str or uproot4.reading.ReadOnlyDirectory
                The filename or already opened file using e.g. ``uproot4.open()``
            treepath : str, optional
                Name of the tree to read in the file
            entry_start : int, optional
                Start at this entry offset in the tree (default 0)
            entry_stop : int, optional
                Stop at this entry offset in the tree (default end of tree)
            runtime_cache : dict, optional
                A dict-like interface to a cache object. This cache is expected to last the
                duration of the program only, and will be used to hold references to materialized
                awkward1 arrays, etc.
            persistent_cache : dict, optional
                A dict-like interface to a cache object. Only bare numpy arrays will be placed in this cache,
                using globally-unique keys.
            schemaclass : BaseSchema
                A schema class deriving from `BaseSchema` and implementing the desired view of the file
            metadata : dict, optional
                Arbitrary metadata to add to the `base.NanoEvents` object
        """
        if not issubclass(schemaclass, BaseSchema):
            raise RuntimeError("Invalid schema type")
        if isinstance(file, str):
            tree = uproot4.open(file + ":" + treepath)
        elif isinstance(file, uproot4.reading.ReadOnlyDirectory):
            tree = file[treepath]
        if entry_start is None or entry_start < 0:
            entry_start = 0
        if entry_stop is None or entry_stop > tree.num_entries:
            entry_stop = tree.num_entries
        partition_tuple = (
            str(tree.file.uuid),
            tree.object_path,
            "{0}-{1}".format(entry_start, entry_stop),
        )
        uuidpfn = {partition_tuple[0]: tree.file.file_path}
        mapping = UprootSourceMapping(uuidpfn)
        mapping.preload_tree(partition_tuple[0], partition_tuple[1], tree)
        if persistent_cache is not None:
            mapping = CachedMapping(persistent_cache, mapping)
        base_form = cls._extract_base_form(tree)
        if metadata is not None:
            base_form["parameters"]["metadata"] = metadata
        schema = schemaclass(base_form)
        return cls(schema, mapping, tuple_to_key(partition_tuple), cache=runtime_cache)

    def __len__(self):
        uuid, treepath, entryrange = key_to_tuple(self._partition_key)
        start, stop = (int(x) for x in entryrange.split("-"))
        return stop - start

    @classmethod
    def _extract_base_form(cls, tree):
        branch_forms = {}
        for key, branch in tree.iteritems():
            if "," in key or "!" in key:
                warnings.warn(
                    f"Skipping {key} because it contains characters that NanoEvents cannot accept [,!]"
                )
                continue
            if len(branch):
                continue
            form = branch.interpretation.awkward_form(None)
            form = uproot4._util.awkward_form_remove_uproot(awkward1, form)
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

    def events(self):
        """Build events

        """
        if self._events is None:
            behavior = dict(self._schema.behavior)
            behavior["__events_factory__"] = self
            self._events = awkward1.from_arrayset(
                self._schema.form,
                self._mapping,
                prefix=self._partition_key,
                sep="/",
                lazy=True,
                lazy_lengths=len(self),
                lazy_cache="attach" if self._cache is None else self._cache,
                behavior=behavior,
            )

        return self._events
