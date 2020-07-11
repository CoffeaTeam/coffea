from weakref import WeakValueDictionary
import numpy
import awkward1
import uproot


def _with_length(array: awkward1.layout.VirtualArray, length: int):
    return awkward1.layout.VirtualArray(
        array.generator.with_length(length),
        array.cache,
        array.cache_key,
        array.identities,
        array.parameters,
    )


class NanoEventsFactory:
    default_mixins = {
        "GenPart": "GenParticle",
        "Electron": "Electron",
        "Muon": "Muon",
        "Tau": "Tau",
        "Photon": "Photon",
        "Jet": "Jet",
        "FatJet": "FatJet",
    }
    _active = WeakValueDictionary()

    def __init__(
        self,
        file,
        treename="Events",
        entrystart=None,
        entrystop=None,
        cache=None,
        mixin_map=None,
        metadata=None,
    ):
        if not isinstance(file, uproot.rootio.ROOTDirectory):
            file = uproot.open(file)
        self._tree = file[treename]
        self._entrystart, self._entrystop = uproot.tree._normalize_entrystartstop(
            self._tree.numentries, entrystart, entrystop
        )
        self._keyprefix = "/".join(
            [
                file._context.uuid.hex(),
                treename,
                str(self._entrystart),
                str(self._entrystop),
            ]
        )
        NanoEventsFactory._active[self._keyprefix] = self

        if cache is None:
            cache = awkward1.layout.ArrayCache({})
        else:
            cache = awkward1.layout.ArrayCache(cache)
        self._cache = cache

        self._mixin_map = {}
        self._mixin_map.update(self.default_mixins)
        if mixin_map is not None:
            self._mixin_map.update(mixin_map)

        self._metadata = metadata  # TODO: JSON only?
        self._branches_read = set()
        self._events = None

    @classmethod
    def get_events(cls, key):
        return cls._active[key].events()

    def __len__(self):
        return self._entrystop - self._entrystart

    def reader(self, branch_name, parameters):
        self._branches_read.add(branch_name)
        return awkward1.layout.NumpyArray(
            self._tree[branch_name].array(
                entrystart=self._entrystart, entrystop=self._entrystop, flatten=True
            ),
            parameters=parameters,
        )

    def _array(self, branch_name: bytes):
        interpretation = uproot.interpret(self._tree[branch_name])
        if isinstance(interpretation, uproot.asjagged):
            dtype = interpretation.content.type
            length = None
        else:
            dtype = interpretation.type
            length = len(self)
        parameters = {"__doc__": self._tree[branch_name].title.decode("ascii")}
        # use hint to resolve platform-dependent format
        formhint = awkward1.forms.Form.fromjson('"%s"' % dtype)
        form = awkward1.forms.NumpyForm(
            [], formhint.itemsize, formhint.format, parameters=parameters
        )
        generator = awkward1.layout.ArrayGenerator(
            self.reader, (branch_name, parameters), {}, form=form, length=length,
        )
        return awkward1.layout.VirtualArray(
            generator,
            self._cache,
            cache_key="/".join([self._keyprefix, "file", branch_name.decode("ascii")]),
            parameters=parameters,
        )

    def _listarray(self, counts, content, recordparams):
        offsets = awkward1.layout.Index32(
            numpy.concatenate([[0], numpy.cumsum(counts)])
        )
        length = offsets[-1]
        return awkward1.layout.ListOffsetArray32(
            offsets,
            awkward1.layout.RecordArray(
                {k: _with_length(v, length) for k, v in content.items()},
                parameters=recordparams,
            ),
        )

    def events(self):
        if self._events is not None:
            return self._events

        arrays = {}
        for branch_name in self._tree.keys():
            arrays[branch_name.decode("ascii")] = self._array(branch_name)

        # parse into high-level records (collections, list collections, and singletons)
        collections = set(k.split("_")[0] for k in arrays)
        collections -= set(
            k for k in collections if k.startswith("n") and k[1:] in collections
        )

        def collectionfactory(name):
            mixin = self._mixin_map.get(name, "NanoCollecton")
            if "n" + name in arrays:
                # list collection
                cname = "n" + name
                counts = arrays[cname]
                content = {
                    k[len(cname) :]: arrays[k]
                    for k in arrays
                    if k.startswith(name + "_")
                }
                recordparams = {
                    "__doc__": counts.parameters["__doc__"],
                    "__record__": mixin,
                    "events_key": self._keyprefix,
                }
                form = awkward1.forms.ListOffsetForm(
                    "i32",
                    awkward1.forms.RecordForm(
                        {k: v.form for k, v in content.items()}, parameters=recordparams
                    ),
                )
                generator = awkward1.layout.ArrayGenerator(
                    self._listarray,
                    (counts, content, recordparams),
                    {},
                    form=form,
                    length=len(self),
                )
                return awkward1.layout.VirtualArray(
                    generator,
                    self._cache,
                    cache_key="/".join([self._keyprefix, "file", name]),
                    parameters=recordparams,
                )
            elif name in arrays:
                # singleton
                return arrays[name]
            else:
                # simple collection
                return awkward1.layout.RecordArray(
                    {
                        k[len(name) + 1 :]: arrays[k]
                        for k in arrays
                        if k.startswith(name + "_")
                    },
                    parameters={"__record__": mixin, "events_key": self._keyprefix},
                )

        events = awkward1.layout.RecordArray(
            {name: collectionfactory(name) for name in collections},
            parameters={
                "__record__": "NanoEvents",
                "events_key": self._keyprefix,
                "metadata": self._metadata,
            },
        )

        self._events = awkward1.Array(events)
        return self._events
