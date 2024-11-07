import pytest


def test_processorabc():
    from coffea.processor import ProcessorABC

    class test(ProcessorABC):
        @property
        def accumulator(self):
            pass

        def process(self, df):
            pass

        def postprocess(self, accumulator):
            pass

    with pytest.raises(TypeError):
        proc = ProcessorABC()

    proc = test()

    df = None
    super(test, proc).process(df)

    acc = None
    super(test, proc).postprocess(acc)


def test_mapfilter():
    from functools import partial

    import awkward as ak
    import dask_awkward as dak
    import numpy as np

    from coffea.nanoevents import NanoEventsFactory
    from coffea.processor import mapfilter

    events, report = NanoEventsFactory.from_root(
        {
            "https://github.com/CoffeaTeam/coffea/raw/master/tests/samples/nano_dy.root": "Events"
        },
        metadata={"dataset": "Test"},
        uproot_options={"allow_read_errors_with_report": True},
        steps_per_file=2,
    ).events()

    def process(events):
        # do an emberassing parallel computation
        # only eager awkward is allowed here
        import awkward as ak

        jets = events.Jet
        jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
        return events[ak.num(jets) == 2]

    # check that `mapfilter` only adds 1 layer to the HLG, given that there are already 2 layers from the `NanoEventsFactory` reading
    out = mapfilter(process)(events)
    assert len(out.dask.layers.keys()) == 3

    # check that `mapfilter` can forcefully touch additional columns
    needs = {"events": [("Muon", "pt")]}
    out = mapfilter(process, needs=needs)(events)
    cols = next(iter(dak.necessary_columns(out.Jet.pt).values()))
    assert "Muon_pt" in cols

    # check that `mapfilter` can properly mock output for untraceable computations
    out_like = ak.Array([0.0, 0.0])

    @partial(
        mapfilter,
        needs={"muons": ["pt"]},
        out_like=out_like,
    )
    def untraceable_process(muons):
        # a non-traceable computation for ak.typetracer
        # which needs "pt" column from muons and returns a 1-element array
        return ak.Array([np.sum(ak.to_numpy(muons.pt[0:1]))])

    out = untraceable_process(events.Muon)
    cols = next(iter(dak.necessary_columns(out).values()))
    assert "Muon_pt" in cols
    assert out.compute().typestr == out_like.typestr == "2 * float64"
