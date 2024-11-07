import inspect
import typing as tp
from dataclasses import dataclass

import awkward as ak


@dataclass
class mapfilter:
    """Map a callable across all partitions of any number of collections.
    This decorator is a convenience wrapper around the `dak.map_partitions` function.

    It serves the following purposes:
        - Turn multiple operations into a single node in the Dask graph
        - Explicitly touch columns if necessarily without interacting with the typetracer

    Caveats:
        - The function must use pure eager awkward inside (no delayed operations)
        - The function must return a single argument, i.e. an awkward array
        - The function must be emberassingly parallel

    Parameters
    ----------
    base_fn : Callable
        Function to apply on all partitions, this will get wrapped to
        handle kwargs, including dask collections.
    label : str, optional
        Label for the Dask graph layer; if left to ``None`` (default),
        the name of the function will be used.
    token : str, optional
        Provide an already defined token. If ``None`` a new token will
        be generated.
    meta : Any, optional
        Metadata (typetracer) array for the result (if known). If
        unknown, `fn` will be applied to the metadata of the `args`;
        if that call fails, the first partition of the new collection
        will be used to compute the new metadata **if** the
        ``awkward.compute-known-meta`` configuration setting is
        ``True``. If the configuration setting is ``False``, an empty
        typetracer will be assigned as the metadata.
    output_divisions : int, optional
        If ``None`` (the default), the divisions of the output will be
        assumed unknown. If defined, the output divisions will be
        multiplied by a factor of `output_divisions`. A value of 1
        means constant divisions (e.g. a string based slice). Any
        value greater than 1 means the divisions were expanded by some
        operation. This argument is mainly for internal library
        function implementations.
    traverse : bool
        Unpack basic python containers to find dask collections.
    needs: dict, optional
        If ``None`` (the default), nothing is touched in addition to the
        standard typetracer report. In certain cases, it is necessary to
        touch additional objects **explicitly** to get the correct typetracer report.
        For this, provide a dictionary that maps input argument that's an array to
        the columns/slice of that array that should be touched.
    out_like: tp.Any, optional
        If ``None`` (the default), the output will be computed through the default
        typetracing pass. If a ak.Array is provided, the output will be mocked for the typetracing
        pass as the provided array. This is useful for cases where the output can not be
        computed through the default typetracing pass.


    Returns
    -------
    dask_awkward.Array
        The new collection.

    Examples
    --------
    >>> from coffea.nanoevents import NanoEventsFactory
    >>> from coffea.processor.decorator import mapfilter
    >>> events, report = NanoEventsFactory.from_root(
            {"https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dy.root": "Events"},
            metadata={"dataset": "Test"},
            uproot_options={"allow_read_errors_with_report": True},
            steps_per_file=2,
        ).events()
    >>> @mapfilter
        def process(events):
            # do an emberassing parallel computation
            # only eager awkward is allowed here
            import awkward as ak

            jets = events.Jet
            jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
            return events[ak.num(jets) == 2]
    >>> selected = process(events)
    >>> print(process(events).dask)  # collapsed into a single node (2.)
    HighLevelGraph with 3 layers.
    <dask.highlevelgraph.HighLevelGraph object at 0x11700d640>
    0. from-uproot-0e54dc3659a3c020608e28b03f22b0f4
    1. from-uproot-971b7f00ce02a189422528a5044b08fb
    2. <dask-awkward.lib.core.ArgsKwargsPackedFunction ob-c9ee010d2e5671a2805f6d5106040d55
    >>> print(process.base_fn(events).dask) # call the function as it is (many nodes in the graph); `base_fn` is the function that is wrapped
    HighLevelGraph with 13 layers.
    <dask.highlevelgraph.HighLevelGraph object at 0x136e3d910>
    0. from-uproot-0e54dc3659a3c020608e28b03f22b0f4
    1. from-uproot-971b7f00ce02a189422528a5044b08fb
    2. Jet-efead9353042e606d7ffd59845f4675d
    3. eta-f31547c2a94efc053977790a489779be
    4. absolute-74ced100c5db654eb0edd810542f724a
    5. less-b33e652814e0cd5157b3c0885087edcb
    6. pt-f50c15fa409e60152de61957d2a4a0d8
    7. greater-da496609d36631ac857bb15eba6f0ba6
    8. bitwise-and-a501c0ff0f5bcab618514603d4f78eec
    9. getitem-fc20cad0c32130756d447fc749654d11
    10. <dask-awkward.lib.core.ArgsKwargsPackedFunction ob-0d3090f1c746eafd595782bcacd30d69
    11. equal-a4642445fb4e5da0b852c2813966568a
    12. getitem-f951afb4c4d4b527553f5520f6765e43

    # if you want to touch additional objects explicitly, because they are not touched by the standard typetracer (i.e. due to 'opaque' operations)
    # you can provide a dict of slices that should be touched directly to the decorator, e.g.:
    >>> from functools import partial
    >>> @partial(mapfilter, needs={"events": [("Electron", "pt"), ("Electron", "eta")]})
        def process(events):
            # do an emberassing parallel computation
            # only eager awkward is allowed here
            import awkward as ak

            jets = events.Jet
            jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
            return events[ak.num(jets) == 2]
    >>> selected = process(events)
    >>> print(dak.necessary_columns(selected))
    {'from-uproot-0e54dc3659a3c020608e28b03f22b0f4': frozenset({'Electron_eta', 'Jet_eta', 'nElectron', 'Jet_pt', 'Electron_pt', 'nJet'})}

    """

    base_fn: tp.Callable
    label: str | None = None
    token: str | None = None
    meta: tp.Any | None = None
    output_divisions: int | None = None
    traverse: bool = True
    # additional options that are not available in dak.map_partitions
    needs: dict | None = None
    out_like: ak.Array | None = None

    def wrapped_fn(self, *args: tp.Any, **kwargs: tp.Any):
        ba = inspect.signature(self.base_fn).bind(*args, **kwargs)
        in_arguments = ba.arguments
        if self.needs is not None:
            if not isinstance(self.needs, tp.Mapping):
                msg = "needs argument must be a dictionary mapping of keyword argument that points to awkward arrays to columns that should be touched explicitly."
                raise ValueError(msg)
            tobe_touched = set()
            for arg in self.needs.keys():
                if arg in in_arguments:
                    tobe_touched.add(arg)
                else:
                    msg = f"Argument '{arg}' is not present in the function signature."
                    raise ValueError(msg)
            for arg in tobe_touched:
                array = in_arguments[arg]
                if ak.backend(array) == "typetracer":
                    # touch the objects explicitly
                    for slce in self.needs[arg]:
                        ak.typetracer.touch_data(array[slce])
        if self.out_like is not None:
            # check if we're in the typetracing step
            if any(
                ak.backend(array) == "typetracer" for array in in_arguments.values()
            ):
                # mock the output as the specified type
                if not isinstance(self.out_like, ak.Array):
                    raise ValueError("out_like must be an awkward array")
                if ak.backend(self.out_like) == "typetracer":
                    return self.out_like
                return ak.Array(self.out_like.layout.to_typetracer(forget_length=True))

        return self.base_fn(*args, **kwargs)

    def __call__(self, *args: tp.Any, **kwargs: tp.Any):
        from dask_awkward.lib.core import map_partitions

        return map_partitions(
            self.wrapped_fn,
            *args,
            label=self.label,
            token=self.token,
            meta=self.meta,
            output_divisions=self.output_divisions,
            traverse=self.traverse,
            **kwargs,
        )
