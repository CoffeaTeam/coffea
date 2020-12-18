import warnings
from cachetools import LRUCache
from collections.abc import Mapping
import uproot4
import awkward1
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
            # make sure uproot is single-core since our calling context might not be
            return self.file.read([column_name], use_threads=False)

        # for right now spoof the notion of directories in files
        # parquet can do it but we've gotta convince people to
        # use it like that
        def __getitem__(self, name):
            return self.file

    def __init__(self, uuid_pfnmap, parquet_options={}):
        super(TrivialParquetOpener, self).__init__(uuid_pfnmap)
        self._parquet_options = parquet_options
        self._schema_map = None

    def open_uuid(self, uuid):
        import pyarrow.parquet as pq
        pfn = self._uuid_pfnmap[uuid]
        parfile = pq.ParquetFile(pfn, **self._parquet_options)
        pqmeta = parfile.schema_arrow.metadata
        pquuid = None if pqmeta is None else pqmeta.get(b"uuid", None)
        pfn = None if pqmeta is None else pqmeta.get(b"url", None)
        if str(pquuid) != uuid:
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
        """To save a double-open when using NanoEventsFactory._from_mapping"""
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


def arrow_schema_to_awkward_form(schema):
    import pyarrow as pa
    if isinstance(schema, (pa.lib.ListType, pa.lib.LargeListType)):
        dtype = schema.value_type.to_pandas_dtype()()
        return awkward1.forms.ListOffsetForm(
            offsets="i64",
            content=awkward1.forms.NumpyForm(
                inner_shape=[],
                itemsize=dtype.dtype.itemsize,
                format=dtype.dtype.char,
            ),
        )
    elif isinstance(schema, pa.lib.DataType):
        dtype = schema.to_pandas_dtype()()
        return awkward1.forms.NumpyForm(
            inner_shape=[],
            itemsize=dtype.dtype.itemsize,
            format=dtype.dtype.char,
        )
    else:
        raise Exception("Unrecognized pyarrow array type")
    return None


class ParquetSourceMapping(BaseSourceMapping):
    _debug = False

    class UprootLikeShim:
        def __init__(self, source, column):
            self.source = source
            self.column = column

        def array(self, entry_start, entry_stop):
            import pyarrow as pa
            aspa = self.source.read(self.column)[entry_start:entry_stop][0].chunk(0)
            out = None
            if isinstance(aspa, (pa.lib.ListArray, pa.lib.LargeListArray)):
                value_type = aspa.type.value_type
                offsets = None
                if isinstance(aspa, pa.lib.LargeListArray):
                    offsets = numpy.frombuffer(aspa.buffers()[1], dtype=numpy.int64)[
                        : len(aspa) + 1
                    ]
                else:
                    offsets = numpy.frombuffer(aspa.buffers()[1], dtype=numpy.int32)[
                        : len(aspa) + 1
                    ]
                    offsets = offsets.astype(numpy.int64)
                offsets = awkward1.layout.Index64(offsets)

                if not isinstance(value_type, pa.lib.DataType):
                    raise Exception(
                        "arrow only accepts single jagged arrays for now..."
                    )
                dtype = value_type.to_pandas_dtype()
                flat = aspa.flatten()
                content = numpy.frombuffer(flat.buffers()[1], dtype=dtype)[: len(flat)]
                content = awkward1.layout.NumpyArray(content)
                out = awkward1.layout.ListOffsetArray64(offsets, content)
            elif isinstance(aspa, pa.lib.NumericArray):
                out = numpy.frombuffer(
                    aspa.buffers()[1], dtype=aspa.type.to_pandas_dtype()
                )[: len(aspa)]
                out = awkward1.layout.NumpyArray(out)
            else:
                raise Exception("array is not flat array or jagged list")
            return awkward1.Array(out)

    def __init__(self, fileopener, cache=None, access_log=None):
        super(ParquetSourceMapping, self).__init__(fileopener, cache, access_log)

    @classmethod
    def _extract_base_form(cls, arrow_schema):
        column_forms = {}
        for field in arrow_schema:
            key = field.name
            fmeta = field.metadata

            if "," in key or "!" in key:
                warnings.warn(
                    f"Skipping {key} because it contains characters that NanoEvents cannot accept [,!]"
                )
                continue

            form = None
            if b"form" in fmeta:
                form = json.loads(fmeta[b"form"])
            else:
                schema = field.type
                form = arrow_schema_to_awkward_form(schema)
                form = json.loads(form.tojson())

            if (
                form["class"].startswith("ListOffset")
                and form["content"]["class"] == "NumpyArray"  # noqa
            ):
                form["form_key"] = quote(f"{key},!load")
                form["content"]["form_key"] = quote(f"{key},!load,!content")
                if b"title" in fmeta:
                    form["content"]["parameters"] = {
                        "__doc__": fmeta[b"title"].decode()
                    }
                elif "__doc__" not in form["content"]["parameters"]:
                    form["content"]["parameters"] = {"__doc__": key}
            elif form["class"] == "NumpyArray":
                form["form_key"] = quote(f"{key},!load")
                if b"title" in fmeta:
                    form["parameters"] = {"__doc__": fmeta[b"title"].decode()}
                elif "__doc__" not in form["parameters"]:
                    form["parameters"] = {"__doc__": key}
            else:
                warnings.warn(
                    f"Skipping {key} as it is not interpretable by NanoEvents"
                )
                continue
            column_forms[key] = form
        return {
            "class": "RecordArray",
            "contents": column_forms,
            "parameters": {"__doc__": "parquetfile"},
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
