import numpy
import uproot
import awkward
from .methods import collection_methods
from .util import _mixin
from ..util import _hex, _ascii


class NanoCollection(awkward.VirtualArray):
    r'''A NanoAOD collection

    NanoAOD collections are collections of branches formed by name, where:

    - one branch exists named ``name`` and no branches start with ``name_``, interpreted as a single flat array;
    - one branch exists named ``name``, one named ``n{name}``, and no branches start with ``name_``, interpreted as a single jagged array;
    - no branch exists named ``{name}`` and many branches start with ``name_*``, interpreted as a flat table; or
    - one branch exists named ``n{name}`` and many branches start with ``name_*``, interpreted as a jagged table.

    '''
    _mixin_registry = {}

    @classmethod
    def _get_mixin(cls, methods, awkwardtype=None):
        if awkwardtype is None:
            awkwardtype = NanoCollection
        if issubclass(awkwardtype, NanoCollection) and awkwardtype is not NanoCollection:
            raise ValueError("Can only mixin with NanoCollection base or awkward type, not derived")
        key = (methods, awkwardtype)
        if key not in NanoCollection._mixin_registry:
            if methods is None:
                NanoCollection._mixin_registry[key] = awkwardtype
            else:
                NanoCollection._mixin_registry[key] = _mixin(methods, awkwardtype)
        return NanoCollection._mixin_registry[key]

    @classmethod
    def _get_methods(cls):
        for k, v in NanoCollection._mixin_registry.items():
            if v is cls:
                return k[0]
        raise RuntimeError("Unregistered mixin detected")

    @classmethod
    def from_arrays(cls, arrays, name, methods=None):
        '''Build from dictionary of VirtualArrays

        Parameters
        ----------
            arrays : dict
                A mapping from branch name to flat VirtualArray
            name : str
                The name of the collection (see class documentation for interpretation)
            methods : class, optional
                A class deriving from `awkward.array.objects.Methods` that implements additional mixins

        Returns a NanoCollection object, possibly mixed in with methods.
        '''
        outtype = cls._get_mixin(methods)
        jagged = 'n' + name in arrays.keys()
        columns = {k[len(name) + 1:]: arrays[k] for k in arrays.keys() if k.startswith(name + '_')}
        if len(columns) == 0:
            # single-item collection, just forward lazy array
            if name not in arrays.keys():
                raise RuntimeError('Could not find collection %s in dataframe' % name)
            array = outtype.__new__(outtype)
            array.__dict__ = arrays[name].__dict__
            array.__doc__ = arrays[name].__doc__
            if jagged:
                counts = arrays['n' + name]
                out = outtype(
                    outtype._lazyjagged,
                    (name, counts, array, methods),
                    type=awkward.type.ArrayType(len(counts), float('inf'), array.type.to),
                )
                out.__doc__ = counts.__doc__
                array = out
            return array
        elif jagged:
            tabletype = awkward.type.TableType()
            for k, array in columns.items():
                tabletype[k] = array.type.to
            counts = arrays['n' + name]
            out = outtype(
                outtype._lazyjagged,
                (name, counts, columns, methods),
                type=awkward.type.ArrayType(len(counts), float('inf'), tabletype),
            )
            out.__doc__ = counts.__doc__
            return out
        else:
            Table = cls._get_mixin(methods, awkward.Table)
            table = Table.named(name)
            for k, v in columns.items():
                table.contents[k] = v
            return table

    @classmethod
    def _lazyjagged(cls, name, counts, columns, methods=None):
        offsets = awkward.JaggedArray.counts2offsets(counts.array)
        JaggedArray = cls._get_mixin(methods, awkward.JaggedArray)
        Table = cls._get_mixin(methods, awkward.Table)
        if isinstance(columns, dict):
            content = Table.named(name)
            content.type.takes = offsets[-1]
            for k, v in columns.items():
                if not isinstance(v, awkward.VirtualArray):
                    raise RuntimeError
                v.type.takes = offsets[-1]
                content.contents[k] = v
        else:
            if not isinstance(columns, awkward.VirtualArray):
                raise RuntimeError
            columns.type.takes = offsets[-1]
            content = columns
        out = JaggedArray.fromoffsets(offsets, content)
        out.__doc__ = counts.__doc__
        return out

    def _lazy_crossref(self, index, destination):
        # to be used for simple NanoCollection -> NanoCollection index-based mapping
        # e.g. Muon_jetIdx
        if not isinstance(destination, NanoCollection):
            raise ValueError("Destination must be a NanoCollection")
        if not isinstance(destination.array, awkward.JaggedArray):
            raise ValueError("Cross-references imply jagged destination")
        if not isinstance(self.array, awkward.JaggedArray):
            raise NotImplementedError
        IndexedMaskedArray = destination._get_mixin(destination._get_methods(), awkward.IndexedMaskedArray)
        IndexedTable = destination._get_mixin(destination._get_methods(), awkward.Table)
        # repair awkward type now that we've materialized
        index.type.takes = self.array.offsets[-1]
        index = awkward.JaggedArray.fromoffsets(self.array.offsets, content=index)
        globalindex = (index + destination.array.starts).flatten()
        invalid = (index < 0).flatten()
        if any(invalid):
            globalindex[invalid] = -1
            # note: parent virtual must derive from this type
            out = IndexedMaskedArray(
                globalindex,
                destination.array.content,
            )
        else:
            # don't use masked if we don't have to (it has bugs)
            out = IndexedTable.__getitem__(destination.array.content, globalindex)
        # useful for algorithms
        self.array.content['_xref_%s_index' % destination.rowname] = globalindex
        return out

    def _lazy_nested_crossref(self, indices, destination):
        # to be used for stitching a set of indices into a doubly-jagged mapping
        # e.g. Jet_electronIdx1, Jet_electronIdx2
        if not isinstance(destination, NanoCollection):
            raise ValueError("Destination must be a NanoCollection")
        if not isinstance(destination.array, awkward.JaggedArray):
            raise ValueError("Cross-references imply jagged destination")
        if not isinstance(self.array, awkward.JaggedArray):
            raise NotImplementedError
        JaggedArray = destination._get_mixin(destination._get_methods(), awkward.JaggedArray)
        # repair type now that we've materialized
        for idx in indices:
            idx.type.takes = self.array.offsets[-1]
        content = numpy.zeros(len(self.array.content) * len(indices), dtype=awkward.JaggedArray.INDEXTYPE)
        for i, index in enumerate(indices):
            content[i::len(indices)] = numpy.array(index)
        globalindices = awkward.JaggedArray.fromoffsets(
            self.array.offsets,
            awkward.JaggedArray.fromoffsets(
                numpy.arange((len(self.array.content) + 1) * len(indices), step=len(indices)),
                content,
            )
        )
        globalindices = globalindices[globalindices >= 0] + destination.array.starts
        # note: parent virtual must derive from this type
        out = JaggedArray.fromoffsets(
            globalindices.content.offsets,
            content=destination.array.content[globalindices.flatten().flatten()]
        )
        return out

    def _lazy_double_jagged(self, mapping, destination):
        # to be used for nesting a jagged collection into another when a mapping is available
        # e.g. NanoAODJMAR's FatJetPFCands, mapping FatJet -> PFCands
        if not isinstance(destination, NanoCollection):
            raise ValueError("Destination must be a NanoCollection")
        JaggedArray = destination._get_mixin(destination._get_methods(), awkward.JaggedArray)
        # repair awkward type now that we've materialized
        mapping.type.takes = self.array.offsets[-1]
        # technically we materialize the counts of the destination for no reason but OK
        dest_content = destination.array
        if isinstance(dest_content, awkward.JaggedArray):
            dest_content = dest_content.content
        # (otherwise assume it is already flat and ready to go)
        return JaggedArray.fromcounts(mapping.array, dest_content)

    def _getcolumn(self, key):
        name, _, columns, _ = self._args
        if key not in columns:
            # This function is only meant for use in methods' _finalize() while
            # all columns are still virtual. Missing arrays are a sign of an incompatible
            # file or missing preloaded columns. This triggers only if the missing column is accessed.
            def nonexistentarray():
                raise RuntimeError("There was an attempt to read the nonexistent array: %s_%s" % (name, key))
            return awkward.VirtualArray(nonexistentarray)
        return columns[key]

    def __setitem__(self, key, value):
        if self.ismaterialized:
            super(NanoCollection, self).__setitem__(key, value)
        _, _, columns, _ = self._args
        columns[key] = value
        self._type.to.to[key] = value.type.to

    def __delitem__(self, key):
        if self.ismaterialized:
            super(NanoCollection, self).__delitem__(key)
        _, _, columns, _ = self._args
        if key in columns:
            del columns[key]
            del self._type.to.to[key]


class NanoEvents(awkward.Table):
    '''Top-level awkward array representing NanoAOD events chunk

    The default interpretation of the various collections is governed by `coffea.nanoaod.methods.collection_methods`,
    but can optionally be overriden by passing a custom mapping in the various ``from_*`` constructors.
    '''
    @classmethod
    def from_arrays(cls, arrays, methods=None, metadata=None):
        '''Build NanoEvents from a dictionary of arrays

        Parameters
        ----------
            arrays : dict
                A mapping from branch name to flat numpy array or awkward VirtualArray
            methods : dict, optional
                A mapping from collection name to class deriving from `awkward.array.objects.Methods`
                that implements additional mixins
            metadata : dict, optional
                Arbitrary metadata to embed in this NanoEvents table

        Returns a NanoEvents object
        '''
        arrays = dict(arrays)
        for k in arrays:
            if isinstance(arrays[k], awkward.VirtualArray):
                pass
            elif isinstance(arrays[k], numpy.ndarray):
                value = arrays[k]
                arrays[k] = awkward.VirtualArray(lambda: value, type=awkward.type.ArrayType(len(arrays[k]), arrays[k].dtype))
                print(arrays[k])
            else:
                raise ValueError("The array %s : %r is not a valid type" % (k, arrays[k]))
        events = cls.named('event')
        collections = {k.split('_')[0] for k in arrays.keys()}
        collections -= {k for k in collections if k.startswith('n') and k[1:] in collections}
        allmethods = {}
        allmethods.update(collection_methods)
        if methods is not None:
            allmethods.update(methods)
        for name in collections:
            methods = allmethods.get(name, None)
            events.contents[name] = NanoCollection.from_arrays(arrays, name, methods)

        for name in events.columns:
            # soft hasattr via type, to prevent materialization
            if hasattr(type(events[name]), '_finalize'):
                events.contents[name]._finalize(name, events)

        events.metadata = metadata if metadata is not None else {}
        return events

    @classmethod
    def from_file(cls, file, treename=b'Events', entrystart=None, entrystop=None, cache=None, methods=None, metadata=None):
        '''Build NanoEvents directly from ROOT file

        Parameters
        ----------
            file : str or uproot.rootio.ROOTDirectory
                The filename or already opened file using e.g. ``uproot.open()``
            treename : str, optional
                Name of the tree to read in the file, defaults to ``Events``
            entrystart : int, optional
                Start at this entry offset in the tree (default 0)
            entrystop : int, optional
                Stop at this entry offset in the tree (default end of tree)
            cache : dict, optional
                A dict-like interface to a cache object, in which any materialized virtual arrays will be kept
            methods : dict, optional
                A mapping from collection name to class deriving from `awkward.array.objects.Methods`
                that implements custom additional mixins beyond the defaults provided.
            metadata : dict, optional
                Arbitrary metadata to embed in this NanoEvents table

        Returns a NanoEvents object
        '''
        if cache is None:
            cache = {}
        if not isinstance(file, uproot.rootio.ROOTDirectory):
            file = uproot.open(file)
        tree = file[treename]
        entrystart, entrystop = uproot.tree._normalize_entrystartstop(tree.numentries, entrystart, entrystop)
        arrays = {}
        for bname in tree.keys():
            interpretation = uproot.interpret(tree[bname])
            if isinstance(interpretation, uproot.asjagged):
                virtualtype = awkward.type.ArrayType(float('inf'), interpretation.content.type)
            else:
                virtualtype = awkward.type.ArrayType(entrystop - entrystart, interpretation.type)
            array = awkward.VirtualArray(
                tree[bname].array,
                (),
                {'entrystart': entrystart, 'entrystop': entrystop, 'flatten': True},
                type=virtualtype,
                persistentkey=';'.join(str(x) for x in (_hex(file._context.uuid), _ascii(treename), entrystart, entrystop, _ascii(bname))),
                cache=cache,
            )
            array.__doc__ = tree[bname].title.decode('ascii')
            arrays[bname.decode('ascii')] = array
        out = cls.from_arrays(arrays, methods=methods, metadata=metadata)
        out._cache = cache
        return out

    def tolist(self):
        '''Overriden to raise an exception.  Do not call'''
        raise NotImplementedError("NanoEvents cannot be rendered as a list due to cyclic cross-references")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''Overriden to raise an exception.  Do not call'''
        raise NotImplementedError("NanoEvents cannot be broadcast due to cyclic cross-references")

    def __getitem__(self, where):
        '''Overriden to keep metadata sticky'''
        out = super(NanoEvents, self).__getitem__(where)
        if isinstance(out, NanoEvents):
            out.metadata = self.metadata
            if hasattr(self, '_cache'):
                out._cache = self._cache
        return out

    @property
    def materialized(self):
        '''Set of columns or branches that were materialized

        This is only available when constructed from the file constructor
        '''
        if not hasattr(self, '_cache'):
            raise RuntimeError("NanoEvents.materialized only available if constructed with from_file")
        return set(k.split(';')[-1] for k in self._cache.keys())
