import numpy
import uproot
import awkward
from .methods import collection_methods
from .util import _mixin


class NanoCollection(awkward.VirtualArray):
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
    def _lazyflatten(cls, array):
        return array.array.content

    @classmethod
    def from_arrays(cls, arrays, name, methods=None):
        '''
        arrays : dict
            A mapping from branch name to flat VirtualArray
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
        if not isinstance(destination, NanoCollection):
            raise ValueError("Destination must be a NanoCollection")
        if not isinstance(destination.array, awkward.JaggedArray):
            raise ValueError("Cross-references imply jagged destination")
        if not isinstance(self.array, awkward.JaggedArray):
            raise NotImplementedError
        IndexedMaskedArray = destination._get_mixin(destination._get_methods(), awkward.IndexedMaskedArray)
        # repair awkward type now that we've materialized
        index.type.takes = self.array.offsets[-1]
        index = awkward.JaggedArray.fromoffsets(self.array.offsets, content=index)
        globalindex = (index + destination.array.starts).flatten()
        invalid = (index < 0).flatten()
        globalindex[invalid] = -1
        # note: parent virtual must derive from this type
        out = IndexedMaskedArray(
            globalindex,
            destination.array.content,
        )
        # useful for algorithms
        self.array.content['_xref_%s_index' % destination.rowname] = globalindex
        return out

    def _lazy_nested_crossref(self, indices, destination):
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
            JaggedArray.fromoffsets(
                numpy.arange((len(self.array.content) + 1) * len(indices), step=len(indices)),
                content,
            )
        )
        globalindices = globalindices[globalindices >= 0] + destination.array.starts
        # note: parent virtual must derive from this type
        out = globalindices.content.copy(
            content=destination.array.content[globalindices.flatten().flatten()]
        )
        return out

    def _getcolumn(self, key):
        _, _, columns, _ = self._args
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
        del columns[key]
        del self._type.to.to[key]


class NanoEvents(awkward.Table):
    @classmethod
    def from_arrays(cls, arrays, methods=None):
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

        return events

    @classmethod
    def from_file(cls, file, treename=b'Events', entrystart=None, entrystop=None, cache=None, methods=None):
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
                persistentkey=';'.join(str(x) for x in [file._context.uuid.hex(), treename.decode('ascii'), entrystart, entrystop, bname.decode('ascii')]),
                cache=cache,
            )
            array.__doc__ = tree[bname].title
            arrays[bname.decode('ascii')] = array
        return cls.from_arrays(arrays, methods=methods)

    def tolist(self):
        raise NotImplementedError("NanoEvents cannot be rendered as a list due to cyclic cross-references")
