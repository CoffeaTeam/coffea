import os.path as osp
import sys

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


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="problems with paths on windows"
)
def test_lazy_dataframe():
    import uproot

    from coffea.processor import LazyDataFrame

    tree = uproot.open(osp.abspath("tests/samples/nano_dy.root"))["Events"]
    entrystart = 0
    entrystop = 100

    df = LazyDataFrame(tree, entrystart, entrystop, preload_items=["nMuon"])

    assert len(df) == 1

    pt = df["Muon_pt"]
    assert len(df) == 2
    df["Muon_pt_up"] = pt * 1.05
    assert len(df) == 3
    assert "Muon_pt" in df.materialized

    assert "Muon_eta" in df.available

    assert df.size == tree.num_entries

    with pytest.raises(KeyError):
        df["notthere"]


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="problems with paths on windows"
)
def test_lazy_dataframe_getattr():
    import uproot

    from coffea.processor import LazyDataFrame

    tree = uproot.open(osp.abspath("tests/samples/nano_dy.root"))["Events"]
    entrystart = 0
    entrystop = 100

    df = LazyDataFrame(tree, entrystart, entrystop, preload_items=["nMuon"])

    assert len(df) == 1

    df.Muon_pt
    assert len(df) == 2
    assert "Muon_pt" in df.materialized

    assert "Muon_eta" in df.available

    assert df.size == tree.num_entries

    with pytest.raises(AttributeError):
        df.notthere

    import copy

    df2 = copy.copy(df)
    df2.Muon_pt

    with pytest.raises(AttributeError):
        df2.notthere


def test_processor_newaccumulator():
    from coffea.processor import (
        IterativeExecutor,
        ProcessorABC,
        defaultdict_accumulator,
    )

    class Test(ProcessorABC):
        def process(self, item):
            return {"itemsum": item}

        def postprocess(self, accumulator):
            pass

    proc = Test()

    exe = IterativeExecutor()
    out = exe(
        range(10),
        proc.process,
        None,
    )
    assert out == ({"itemsum": 45}, 0)

    class TestOldStyle(ProcessorABC):
        @property
        def accumulator(self):
            return defaultdict_accumulator(int)

        def process(self, item):
            out = self.accumulator.identity()
            out["itemsum"] += item
            return out

        def postprocess(self, accumulator):
            pass

    proc = TestOldStyle()

    exe = IterativeExecutor()
    out = exe(
        range(10),
        proc.process,
        proc.accumulator,
    )
    assert out[0]["itemsum"] == 45
