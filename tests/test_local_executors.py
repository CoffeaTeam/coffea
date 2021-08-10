import sys
import os.path as osp
import pytest
from coffea import processor
from coffea.nanoevents import schemas

if sys.platform.startswith("win"):
    pytest.skip("skipping tests that only function in linux", allow_module_level=True)


@pytest.mark.parametrize("maxchunks", [None, 1000])
@pytest.mark.parametrize("compression", [None, 0, 2])
@pytest.mark.parametrize(
    "executor", [processor.IterativeExecutor, processor.FuturesExecutor]
)
def test_nanoevents_analysis(executor, compression, maxchunks):
    from coffea.processor.test_items import NanoEventsProcessor

    filelist = {
        "DummyBad": {
            "treename": "Events",
            "files": [osp.abspath("tests/samples/non_existent.root")],
        },
        "ZJets": {
            "treename": "Events",
            "files": [osp.abspath("tests/samples/nano_dy.root")],
            "metadata": {"checkusermeta": True, "someusermeta": "hello"},
        },
        "Data": {
            "treename": "Events",
            "files": [osp.abspath("tests/samples/nano_dimuon.root")],
            "metadata": {"checkusermeta": True, "someusermeta2": "world"},
        },
    }
    treename = "Events"

    exe_args = {
        "workers": 1,
        "skipbadfiles": True,
        "schema": processor.NanoAODSchema,
        "compression": compression,
    }

    hists = processor.run_uproot_job(
        filelist,
        treename,
        NanoEventsProcessor(),
        executor,
        executor_args=exe_args,
        maxchunks=maxchunks,
    )

    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66


@pytest.mark.parametrize("chunksize", [100000, 5])
@pytest.mark.parametrize("schema", [None, schemas.BaseSchema])
@pytest.mark.parametrize(
    "executor", [processor.IterativeExecutor, processor.FuturesExecutor]
)
def test_dataframe_analysis(executor, schema, chunksize):
    from coffea.processor.test_items import NanoTestProcessor

    filelist = {
        "ZJets": [osp.abspath("tests/samples/nano_dy.root")],
        "Data": [osp.abspath("tests/samples/nano_dimuon.root")],
    }
    treename = "Events"

    exe_args = {"workers": 1, "schema": schema}

    hists = processor.run_uproot_job(
        filelist,
        treename,
        NanoTestProcessor(),
        executor,
        executor_args=exe_args,
        chunksize=chunksize,
    )

    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66
