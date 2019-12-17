import numpy
import uproot
import awkward
from .methods import collection_methods
from .util import _mixin


class NanoCollection(awkward.VirtualArray):
    @classmethod
    def _lazyflatten(cls, array):
        return array.array.content

    @classmethod
    def from_arrays(cls, arrays, name, methods=None):
        '''
        arrays : dict
            A mapping from branch name to flat VirtualArray
        '''
        jagged = 'n' + name in arrays.keys()
        columns = {k[len(name) + 1:]: arrays[k] for k in arrays.keys() if k.startswith(name + '_')}
        if len(columns) == 0:
            # single-item collection, just forward lazy array
            if name not in arrays.keys():
                raise RuntimeError('Could not find collection %s in dataframe' % name)
            array = arrays[name]
            if methods:
                ArrayType = _mixin(methods, awkward.VirtualArray)
                out = ArrayType.__new__()
                out.__dict__ = array.__dict__
                out.__doc__ = array.__doc__
                array = out
            if jagged:
                counts = arrays['n' + name]
                out = cls(
                    cls._lazyjagged,
                    (name, counts, array, methods),
                    type=awkward.type.ArrayType(len(counts), float('inf'), array.type.to),
                )
                out.__doc__ = counts.__doc__
                array = out
            return array
        elif jagged:
            if methods:
                cls = _mixin(methods, cls)
            tabletype = awkward.type.TableType()
            for k, array in columns.items():
                tabletype[k] = array.type.to
            counts = arrays['n' + name]
            out = cls(
                cls._lazyjagged,
                (name, counts, columns, methods),
                type=awkward.type.ArrayType(len(counts), float('inf'), tabletype),
            )
            out.__doc__ = counts.__doc__
            return out
        else:
            if methods is None:
                Table = awkward.Table
            else:
                Table = _mixin(methods, awkward.Table)
            table = Table.named(name)
            for k, v in columns.items():
                table.contents[k] = v
            return table

    @classmethod
    def _lazyjagged(cls, name, counts, columns, methods=None):
        offsets = awkward.JaggedArray.counts2offsets(counts.array)
        if methods is None:
            JaggedArray = awkward.JaggedArray
            Table = awkward.Table
        else:
            JaggedArray = _mixin(methods, awkward.JaggedArray)
            Table = _mixin(methods, awkward.Table)
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
        if not isinstance(destination.array, awkward.JaggedArray):
            raise RuntimeError
        if not isinstance(self.array, awkward.JaggedArray):
            raise NotImplementedError
        # repair type now that we've materialized
        index.type.takes = self.array.offsets[-1]
        index = awkward.JaggedArray.fromoffsets(self.array.offsets, content=index)
        globalindex = (index + destination.array.starts).flatten()
        invalid = (index < 0).flatten()
        globalindex[invalid] = -1
        # note: parent virtual must derive from this type
        out = awkward.IndexedMaskedArray(
            globalindex,
            destination.array.content,
        )
        # useful for algorithms
        self.array.content['_xref_%s_index' % destination.rowname] = globalindex
        return out

    def _lazy_nested_crossref(self, indices, destination):
        if not isinstance(destination.array, awkward.JaggedArray):
            raise RuntimeError
        if not isinstance(self.array, awkward.JaggedArray):
            raise NotImplementedError
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
    def from_arrays(cls, arrays, methods={}):
        events = cls.named('event')
        collections = {k.split('_')[0] for k in arrays.keys()}
        collections -= {k for k in collections if k.startswith('n') and k[1:] in collections}
        allmethods = {}
        allmethods.update(collection_methods)
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
    def from_file(cls, file, treename=b'Events', entrystart=None, entrystop=None, cache=None):
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
        return cls.from_arrays(arrays)
