import warnings
import weakref
import awkward1
import uproot4 as uproot
import pathlib
import io
import uuid

from coffea.nanoevents.util import quote, key_to_tuple, tuple_to_key
from coffea.nanoevents.mapping import (
    TrivialUprootOpener,
    TrivialParquetOpener,
    UprootSourceMapping,
    ParquetSourceMapping,
    CachedMapping,
)
from coffea.nanoevents.schemas import BaseSchema, NanoAODSchema


class NanoEventsFactory:
    """A factory class to build NanoEvents objects"""

    def __init__(self, schema, mapping, partition_key, cache=None):
        self._schema = schema
        self._mapping = mapping
        self._partition_key = partition_key
        self._cache = cache
        self._events = lambda: None

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
        self._events = lambda: None

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
        uproot_options={},
        access_log=None,
    ):
        """Quickly build NanoEvents from a root file

        Parameters
        ----------
            file : str or uproot.reading.ReadOnlyDirectory
                The filename or already opened file using e.g. ``uproot.open()``
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
            uproot_options : dict, optional
                Any options to pass to ``uproot.open``
            access_log : list, optional
                Pass a list instance to record which branches were lazily accessed by this instance
        """
        warnings.warn(
            "DEPRECATION NOTICE:\nNanoEventsFactory.from_file is deprecated, please"
            " migrate your code to use NanoEventsFactory.from_root.\nUsing"
            " NanoEventsFactory.from_file will result in an error in the next"
            " major release of coffea."
        )
        return cls.from_root(
            file,
            treepath,
            entry_start,
            entry_stop,
            runtime_cache,
            persistent_cache,
            schemaclass,
            metadata,
            uproot_options,
            access_log,
        )

    @classmethod
    def from_root(
        cls,
        file,
        treepath="/Events",
        entry_start=None,
        entry_stop=None,
        runtime_cache=None,
        persistent_cache=None,
        schemaclass=NanoAODSchema,
        metadata=None,
        uproot_options={},
        access_log=None,
    ):
        """Quickly build NanoEvents from a root file

        Parameters
        ----------
            file : str or uproot.reading.ReadOnlyDirectory
                The filename or already opened file using e.g. ``uproot.open()``
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
            uproot_options : dict, optional
                Any options to pass to ``uproot.open``
            access_log : list, optional
                Pass a list instance to record which branches were lazily accessed by this instance
        """
        if isinstance(file, str):
            tree = uproot.open(file, **uproot_options)[treepath]
        elif isinstance(file, uproot.reading.ReadOnlyDirectory):
            tree = file[treepath]
        elif "<class 'uproot.rootio.ROOTDirectory'>" == str(type(file)):
            raise RuntimeError(
                "The file instance (%r) is an uproot3 type, but this module is only compatible with uproot4 or higher"
                % file
            )
        else:
            raise TypeError("Invalid file type (%s)" % (str(type(file))))

        if entry_start is None or entry_start < 0:
            entry_start = 0
        if entry_stop is None or entry_stop > tree.num_entries:
            entry_stop = tree.num_entries

        partition_key = (
            str(tree.file.uuid),
            tree.object_path,
            "{0}-{1}".format(entry_start, entry_stop),
        )
        uuidpfn = {partition_key[0]: tree.file.file_path}
        mapping = UprootSourceMapping(
            TrivialUprootOpener(uuidpfn, uproot_options), access_log=access_log
        )
        mapping.preload_column_source(partition_key[0], partition_key[1], tree)

        base_form = mapping._extract_base_form(tree)

        return cls._from_mapping(
            mapping,
            partition_key,
            base_form,
            runtime_cache,
            persistent_cache,
            schemaclass,
            metadata,
        )

    @classmethod
    def from_parquet(
        cls,
        file,
        treepath="/Events",
        entry_start=None,
        entry_stop=None,
        runtime_cache=None,
        persistent_cache=None,
        schemaclass=NanoAODSchema,
        metadata=None,
        parquet_options={},
        access_log=None,
    ):
        """Quickly build NanoEvents from a parquet file

        Parameters
        ----------
            file : str, pathlib.Path, pyarrow.NativeFile, or python file-like
                The filename or already opened file using e.g. ``uproot.open()``
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
            parquet_options : dict, optional
                Any options to pass to ``pyarrow.parquet.ParquetFile``
            access_log : list, optional
                Pass a list instance to record which branches were lazily accessed by this instance
        """
        import pyarrow
        import pyarrow.parquet
        ftypes = (
            str,
            pathlib.Path,
            pyarrow.NativeFile,
            io.TextIOBase,
            io.BufferedIOBase,
            io.RawIOBase,
            io.IOBase,
        )
        if isinstance(file, ftypes):
            table_file = pyarrow.parquet.ParquetFile(file, **parquet_options)
        elif isinstance(file, pyarrow.parquet.ParquetFile):
            table_file = file
        else:
            raise TypeError("Invalid file type (%s)" % (str(type(file))))

        if entry_start is None or entry_start < 0:
            entry_start = 0
        if entry_stop is None or entry_stop > table_file.metadata.num_rows:
            entry_stop = table_file.metadata.num_rows

        pqmeta = table_file.schema_arrow.metadata
        pquuid = None if pqmeta is None else pqmeta.get(b"uuid", None)
        pqobj_path = None if pqmeta is None else pqmeta.get(b"object_path", None)

        partition_key = (
            str(None) if pquuid is None else pquuid.decode("ascii"),
            str(None) if pqobj_path is None else pqobj_path.decode("ascii"),
            "{0}-{1}".format(entry_start, entry_stop),
        )
        uuidpfn = {partition_key[0]: pqobj_path}
        mapping = ParquetSourceMapping(
            TrivialParquetOpener(uuidpfn, parquet_options), access_log=access_log
        )
        shim = TrivialParquetOpener.UprootLikeShim(table_file)
        mapping.preload_column_source(partition_key[0], partition_key[1], shim)

        base_form = mapping._extract_base_form(table_file.schema_arrow)

        return cls._from_mapping(
            mapping,
            partition_key,
            base_form,
            runtime_cache,
            persistent_cache,
            schemaclass,
            metadata,
        )

    @classmethod
    def _from_mapping(
        cls,
        mapping,
        partition_key,
        base_form,
        runtime_cache,
        persistent_cache,
        schemaclass,
        metadata,
    ):
        """Quickly build NanoEvents from a root file

        Parameters
        ----------
            mapping : Mapping
                The mapping of a column_source to columns.
            partition_key : tuple
                Basic information about the column source, uuid, paths.
            base_form : dict
                The awkward form describing the nanoevents interpretation of the mapped file.
            runtime_cache : dict
                A dict-like interface to a cache object. This cache is expected to last the
                duration of the program only, and will be used to hold references to materialized
                awkward1 arrays, etc.
            persistent_cache : dict
                A dict-like interface to a cache object. Only bare numpy arrays will be placed in this cache,
                using globally-unique keys.
            schemaclass : BaseSchema
                A schema class deriving from `BaseSchema` and implementing the desired view of the file
            metadata : dict
                Arbitrary metadata to add to the `base.NanoEvents` object

        """
        if not issubclass(schemaclass, BaseSchema):
            raise RuntimeError("Invalid schema type")
        if persistent_cache is not None:
            mapping = CachedMapping(persistent_cache, mapping)
        if metadata is not None:
            base_form["parameters"]["metadata"] = metadata
        schema = schemaclass(base_form)
        return cls(schema, mapping, tuple_to_key(partition_key), cache=runtime_cache)

    def __len__(self):
        uuid, treepath, entryrange = key_to_tuple(self._partition_key)
        start, stop = (int(x) for x in entryrange.split("-"))
        return stop - start

    def events(self):
        """Build events"""
        events = self._events()
        if events is None:
            behavior = dict(self._schema.behavior)
            behavior["__events_factory__"] = self
            events = awkward1.from_arrayset(
                self._schema.form,
                self._mapping,
                prefix=self._partition_key,
                sep="/",
                lazy=True,
                lazy_lengths=len(self),
                lazy_cache="new" if self._cache is None else self._cache,
                behavior=behavior,
            )
            self._events = weakref.ref(events)

        return events
