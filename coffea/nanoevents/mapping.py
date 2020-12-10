import warnings
from cachetools import LRUCache
from collections.abc import Mapping
import uproot4
import awkward1
import pyarrow
import numpy
import json
from coffea.nanoevents import transforms
from coffea.nanoevents.util import quote, key_to_tuple, tuple_to_key
from abc import abstractmethod


class UUIDOpener:
    def __init__(self, uuid_pfnmap):
        self._uuid_pfnmap = uuid_pfnmap

    @abstractmethod
    def open_uuid(self, uuid):
        pass


class TrivialUprootOpener(UUIDOpener):
    def __init__(self, uuid_pfnmap, uproot_options={}):
        super(TrivialUprootOpener, self).__init__(uuid_pfnmap)
        self._uproot_options = uproot_options

    def open_uuid(self, uuid):
        pfn = self._uuid_pfnmap[uuid]
        rootdir = uproot4.open(pfn, **self._uproot_options)
        if str(rootdir.file.uuid) != uuid:
            raise RuntimeError(
                f"UUID of file {pfn} does not match expected value ({uuid})"
            )
        return rootdir


# IMPORTANT -> For now the uuid is just the uuid of the pfn.
#              Later we should use the ParquetFile common_metadata to populate.
class TrivialParquetOpener(UUIDOpener):

    class UprootLikeShim:
        def __init__(self, file):
            self.file = file

        def read(self, column_name):
            return self.file.read([column_name])

        # for right now spoof the notion of directories in files
        # parquet can do it but we've gotta convince people to
        # use it like that
        def __getitem__(self, name):
            return self.file

    def __init__(self, uuid_pfnmap, parquet_options={}):
        super(TrivialUprootOpener, self).__init__(uuid_pfnmap)
        self._parquet_options = parquet_options
        self._schema_map = None

    def open_uuid(self, uuid):
        pfn = self._uuid_pfnmap[uuid]
        parfile = pyarrow.parquet.ParquetFile(pfn, **self._parquet_options)
        if parfile.common_metadata is not None and 'uuid' in parfile.common_metadata:
            uuid = parfile.common_metadata['uuid']
            if str(parfile.file.uuid) != uuid:
                raise RuntimeError(
                    f"UUID of file {pfn} does not match expected value ({uuid})"
                )
        return TrivialParquetOpener.UprootLikeShim(parfile)


class BaseSourceMapping(Mapping):
    _debug = False

    def __init__(self, fileopener, cache=None, access_log=None):
        self._fileopener = fileopener
        self._cache = cache
        self._access_log = access_log
        self.setup()

    def setup(self):
        if self._cache is None:
            self._cache = LRUCache(1)

    @classmethod
    @abstractmethod
    def _extract_base_form(cls, source):
        pass

    def __getstate__(self):
        return {
            "fileopener": self._fileopener,
        }

    def __setstate__(self, state):
        self._fileopener = state["fileopener"]
        self._cache = None
        self.setup()

    def _column_source(self, uuid, path_in_source):
        key = self.key_root() + tuple_to_key((uuid, path_in_source))
        try:
            return self._cache[key]
        except KeyError:
            pass
        source = self._fileopener.open_uuid(uuid)[path_in_source]
        self._cache[key] = source
        return source

    def preload_column_source(self, uuid, path_in_source, source):
        """To save a double-open when using NanoEventsFactory.from_file"""
        key = self.key_root() + tuple_to_key((uuid, path_in_source))
        self._cache[key] = source

    @abstractmethod
    def get_column_handle(self, columnsource, name):
        pass

    @abstractmethod
    def extract_column(self, columnhandle):
        pass

    @classmethod
    def interpret_key(cls, key):
        uuid, treepath, entryrange, form_key, *layoutattr = key_to_tuple(key)
        start, stop = (int(x) for x in entryrange.split("-"))
        nodes = form_key.split(",")
        if len(layoutattr) == 1:
            nodes.append("!" + layoutattr[0])
        elif len(layoutattr) > 1:
            raise RuntimeError(f"Malformed key: {key}")
        return uuid, treepath, start, stop, nodes

    def __getitem__(self, key):
        uuid, treepath, start, stop, nodes = UprootSourceMapping.interpret_key(key)
        if UprootSourceMapping._debug:
            print("Gettting:", uuid, treepath, start, stop, nodes)
        stack = []
        skip = False
        for node in nodes:
            if skip:
                skip = False
                continue
            elif node == "!skip":
                skip = True
                continue
            elif node == "!load":
                handle_name = stack.pop()
                if self._access_log is not None:
                    self._access_log.append(handle_name)
                handle = self.get_column_handle(
                    self._column_source(uuid, treepath), handle_name
                )
                stack.append(self.extract_column(handle, start, stop))
            elif node.startswith("!"):
                tname = node[1:]
                if not hasattr(transforms, tname):
                    raise RuntimeError(
                        f"Syntax error in form_key: no transform named {tname}"
                    )
                getattr(transforms, tname)(stack)
            else:
                stack.append(node)
        if len(stack) != 1:
            raise RuntimeError(f"Syntax error in form key {nodes}")
        out = stack.pop()
        try:
            out = numpy.array(out)
        except ValueError:
            if UprootSourceMapping._debug:
                print(out)
            raise RuntimeError(
                f"Left with non-bare array after evaluating form key {nodes}"
            )
        return out

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class UprootSourceMapping(BaseSourceMapping):
    _debug = False

    def __init__(self, fileopener, cache=None, access_log=None):
        super(UprootSourceMapping, self).__init__(fileopener, cache, access_log)

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
            decompression_executor=uproot4.source.futures.TrivialExecutor(),
            interpretation_executor=uproot4.source.futures.TrivialExecutor(),
        )

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class ParquetSourceMapping(BaseSourceMapping):
    _debug = False

    class UprootLikeShim:
        def __init__(self, source, column):
            self.source = source
            self.column = column

        def array(self, entry_start, entry_stop):
            return awkward1.from_arrow(self.source.read(self.column)[entry_start:entry_stop][0])

    def __init__(self, fileopener, cache=None, access_log=None):
        super(ParquetSourceMapping, self).__init__(fileopener, cache, access_log)

    @classmethod
    def _extract_base_form(cls, source):
        column_forms = {}
        for key, column in source.iteritems():
            if "," in key or "!" in key:
                warnings.warn(
                    f"Skipping {key} because it contains characters that NanoEvents cannot accept [,!]"
                )
                continue
            if len(column):
                continue
            form = column.interpretation.awkward_form(None)
            form = uproot4._util.awkward_form_remove_uproot(awkward1, form)
            form = json.loads(form.tojson())
            if (
                form["class"].startswith("ListOffset")
                and form["content"]["class"] == "NumpyArray"  # noqa
            ):
                form["form_key"] = quote(f"{key},!load")
                form["content"]["form_key"] = quote(f"{key},!load,!content")
                form["content"]["parameters"] = {"__doc__": column.title}
            elif form["class"] == "NumpyArray":
                form["form_key"] = quote(f"{key},!load")
                form["parameters"] = {"__doc__": column.title}
            else:
                warnings.warn(
                    f"Skipping {key} as it is not interpretable by NanoEvents"
                )
                continue
            column_forms[key] = form

        return {
            "class": "RecordArray",
            "contents": column_forms,
            "parameters": {"__doc__": source.title},
            "form_key": "",
        }

    def key_root(self):
        return "ParquetSourceMapping:"

    def preload_column_source(self, uuid, path_in_source, source):
        """To save a double-open when using NanoEventsFactory.from_file"""
        key = self.key_root() + tuple_to_key((uuid, path_in_source))
        self._cache[key] = source

    def get_column_handle(self, columnsource, name):
        return ParquetSourceMapping.UprootLikeShim(columnsource, name)

    def extract_column(self, columnhandle, start, stop):
        # make sure uproot is single-core since our calling context might not be
        return columnhandle.array(entry_start=start, entry_stop=stop)

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError


class CachedMapping(Mapping):
    """A cache-wrapped mapping
    Reads will call into ``cache`` first, and if no key exists,
    the read will fall back to ``base``, saving the reult into ``cache``.
    """

    def __init__(self, cache, base):
        self.cache = cache
        self.base = base
        self.stats = {"hit": 0, "miss": 0}

    def __getitem__(self, key):
        try:
            value = self.cache[key]
            self.stats["hit"] += 1
            return value
        except KeyError:
            value = self.base[key]
            self.cache[key] = value
            self.stats["miss"] += 1
            return value

    def __iter__(self):
        return iter(self.base)

    def __len__(self):
        return len(self.base)
