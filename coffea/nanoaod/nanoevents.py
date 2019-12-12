import numpy
import awkward
from .methods import collection_methods
from .util import _mixin


class NanoCollection(awkward.VirtualArray):
    _already_flat = False

    @classmethod
    def _lazyflatten(cls, array):
        return array.array.content

    @classmethod
    def from_arrays(cls, arrays, name, methods=None):
        '''
        arrays : object
            An object with attributes: columns, __len__, and __getitem__
            where the latter returns virtual arrays or virtual jagged arrays
        '''
        jagged = 'n' + name in arrays.columns
        columns = {k[len(name) + 1:]: arrays[k] for k in arrays.columns if k.startswith(name + '_')}
        if len(columns) == 0:
            # single-item collection, just forward lazy array (possibly jagged)
            if name not in arrays.columns:
                raise RuntimeError('Could not find collection %s in dataframe' % name)
            if methods:
                ArrayType = _mixin(methods, type(arrays[name]))
                return ArrayType(arrays[name])
            return arrays[name]
        elif not jagged:
            if methods is None:
                Table = awkward.Table
            else:
                Table = _mixin(methods, awkward.Table)
            table = Table.named(name)
            for k, v in columns.items():
                table[k] = v
            return table
        else:  # jagged
            if methods:
                cls = _mixin(methods, cls)
            tabletype = awkward.type.TableType()
            for k, array in columns.items():
                tabletype[k] = array.type.to.to
            counts = arrays['n' + name]
            out = cls(
                cls._lazyjagged,
                (name, counts, columns, methods),
                type=awkward.type.ArrayType(len(arrays), float('inf'), tabletype),
            )
            out.__doc__ = counts.__doc__
            return out

    @classmethod
    def _lazyjagged(cls, name, counts, columns, methods=None):
        offsets = awkward.JaggedArray.counts2offsets(counts.array)
        if methods is None:
            JaggedArray = awkward.JaggedArray
            Table = awkward.Table
        else:
            JaggedArray = _mixin(methods, awkward.JaggedArray)
            Table = _mixin(methods, awkward.Table)
        table = Table.named(name)
        for k, v in columns.items():
            if not isinstance(v, awkward.VirtualArray):
                raise RuntimeError
            if isinstance(v, NanoCollection) and v._already_flat:
                v.type.takes = offsets[-1]
                table[k] = v
            else:
                col = type(v)(NanoCollection._lazyflatten, (v,), type=awkward.type.ArrayType(offsets[-1], v.type.to.to))
                col.__doc__ = v.__doc__
                table[k] = col
        out = JaggedArray.fromoffsets(offsets, table)
        out.__doc__ = counts.__doc__
        return out

    def _lazy_crossref(self, index, destination):
        if not isinstance(destination.array, awkward.JaggedArray):
            raise RuntimeError
        if not isinstance(self.array, awkward.JaggedArray):
            raise NotImplementedError
        globalindex = (index + destination.array.starts).flatten()
        invalid = (index < 0).flatten()
        globalindex[invalid] = -1
        # note: parent virtual must derive from this type and have _already_flat = True
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
        content = numpy.zeros(len(self.array.content) * len(indices), dtype=awkward.JaggedArray.INDEXTYPE)
        for i, index in enumerate(indices):
            content[i::len(indices)] = index.flatten()
        globalindices = awkward.JaggedArray.fromoffsets(
            self.array.offsets,
            awkward.JaggedArray.fromoffsets(
                numpy.arange((len(self.array.content) + 1) * len(indices), step=len(indices)),
                content,
            )
        )
        globalindices = globalindices[globalindices >= 0] + destination.array.starts
        # note: parent virtual must derive from this type and have _already_flat = True
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
        if isinstance(value, NanoCollection) and value._already_flat:
            self._type.to.to[key] = value.type.to
        else:
            self._type.to.to[key] = value.type.to.to

    def __delitem__(self, key):
        if self.ismaterialized:
            super(NanoCollection, self).__delitem__(key)
        _, _, columns, _ = self._args
        del columns[key]
        del self._type.to.to[key]


class NanoEvents(awkward.Table):
    @classmethod
    def from_arrays(cls, arrays, collection_methods_overrides={}):
        events = cls.named('event')
        collections = {k.split('_')[0] for k in arrays.columns}
        collections -= {k for k in collections if k.startswith('n') and k[1:] in collections}
        allmethods = {}
        allmethods.update(collection_methods)
        allmethods.update(collection_methods_overrides)
        for name in collections:
            methods = allmethods.get(name, None)
            events[name] = NanoCollection.from_arrays(arrays, name, methods)

        # finalize
        del events.Photon['mass']

        parent_type = awkward.type.ArrayType(float('inf'), awkward.type.OptionType(events.GenPart.type.to.to))
        parent_type.check = False  # break recursion
        gen_parent = type(events.GenPart)(
            events.GenPart._lazy_crossref,
            args=(events.GenPart._getcolumn('genPartIdxMother'), events.GenPart),
            type=parent_type,
        )
        gen_parent._already_flat = True
        gen_parent.__doc__ = events.GenPart.__doc__
        events.GenPart['parent'] = gen_parent
        child_type = awkward.type.ArrayType(float('inf'), float('inf'), events.GenPart.type.to.to)
        child_type.check = False
        children = type(events.GenPart)(
            events.GenPart._lazy_findchildren,
            args=(events.GenPart._getcolumn('genPartIdxMother'),),
            type=child_type,
        )
        children._already_flat = True
        children.__doc__ = events.GenPart.__doc__
        events.GenPart['children'] = children
        del events.GenPart['genPartIdxMother']
        # now that we've created a monster, turn off the safety switch
        events.GenPart.type.check = False

        embedded_subjets = type(events.SubJet)(
            events.FatJet._lazy_nested_crossref,
            args=([events.FatJet._getcolumn('subJetIdx1'), events.FatJet._getcolumn('subJetIdx2')], events.SubJet),
            type=awkward.type.ArrayType(float('inf'), float('inf'), events.SubJet.type.to.to),
        )
        embedded_subjets._already_flat = True
        embedded_subjets.__doc__ = events.SubJet.__doc__
        events.FatJet['subjets'] = embedded_subjets
        del events.FatJet['subJetIdx1']
        del events.FatJet['subJetIdx2']

        return events
