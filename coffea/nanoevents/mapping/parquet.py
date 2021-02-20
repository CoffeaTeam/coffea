import warnings
from cachetools import LRUCache
from collections.abc import Mapping
import awkward
import numpy
import json
from coffea.nanoevents.mapping.base import UUIDOpener, BaseSourceMapping
from coffea.nanoevents.util import quote, key_to_tuple, tuple_to_key
import pyarrow.dataset as ds


# IMPORTANT -> For now the uuid is just the uuid of the pfn.
#              Later we should use the ParquetFile common_metadata to populate.
class TrivialParquetOpener(UUIDOpener):
    class UprootLikeShim:
        def __init__(self, file, dataset):
            self.file = file
            self.dataset = dataset

        def read(self, column_name):
            # make sure uproot is single-core since our calling context might not be
            return self.dataset.to_table(use_threads=False, columns=[column_name])

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


def arrow_schema_to_awkward_form(schema):
    import pyarrow as pa

    if isinstance(schema, (pa.lib.ListType, pa.lib.LargeListType)):
        dtype = schema.value_type.to_pandas_dtype()()
        return awkward.forms.ListOffsetForm(
            offsets="i64",
            content=awkward.forms.NumpyForm(
                inner_shape=[],
                itemsize=dtype.dtype.itemsize,
                format=dtype.dtype.char,
            ),
        )
    elif isinstance(schema, pa.lib.DataType):
        dtype = schema.to_pandas_dtype()()
        return awkward.forms.NumpyForm(
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
                offsets = awkward.layout.Index64(offsets)

                if not isinstance(value_type, pa.lib.DataType):
                    raise Exception(
                        "arrow only accepts single jagged arrays for now..."
                    )
                dtype = value_type.to_pandas_dtype()
                flat = aspa.flatten()
                content = numpy.frombuffer(flat.buffers()[1], dtype=dtype)[: len(flat)]
                content = awkward.layout.NumpyArray(content)
                out = awkward.layout.ListOffsetArray64(offsets, content)
            elif isinstance(aspa, pa.lib.NumericArray):
                out = numpy.frombuffer(
                    aspa.buffers()[1], dtype=aspa.type.to_pandas_dtype()
                )[: len(aspa)]
                out = awkward.layout.NumpyArray(out)
            else:
                raise Exception("array is not flat array or jagged list")
            return awkward.Array(out)

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
