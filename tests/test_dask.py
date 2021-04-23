from __future__ import print_function, division
from coffea import processor

import pytest


def do_dask_job(client, filelist, compression=0):
    treename = "Events"
    from coffea.processor.test_items import NanoTestProcessor

    proc = NanoTestProcessor()
    exe_args = {
        "client": client,
        "compression": compression,
    }
    hists = processor.run_uproot_job(
        filelist,
        treename,
        processor_instance=proc,
        executor=processor.dask_executor,
        executor_args=exe_args,
    )

    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66


def do_dask_cached(client, filelist, cachestrategy=None):
    from coffea.nanoevents import NanoAODSchema
    from coffea.processor.test_items import NanoEventsProcessor
    from coffea.processor.dask import register_columncache

    register_columncache(client)

    exe_args = {
        "client": client,
        "schema": NanoAODSchema,
        "cachestrategy": cachestrategy,
        "savemetrics": True,
        "worker_affinity": True if cachestrategy is not None else False,
    }
    hists, metrics = processor.run_uproot_job(
        filelist,
        "Events",
        processor_instance=NanoEventsProcessor(
            canaries=[
                "a9490124-3648-11ea-89e9-f5b55c90beef/%2FEvents%3B1/0-40/nMuon%2C%21load%2C%21counts2offsets%2C%21skip/offsets",
                "a9490124-3648-11ea-89e9-f5b55c90beef/%2FEvents%3B1/0-40/Muon_phi%2C%21load%2C%21content",
                "a9490124-3648-11ea-89e9-f5b55c90beef/%2FEvents%3B1/0-40/Muon_pt%2C%21load%2C%21content",
                "a9490124-3648-11ea-89e9-f5b55c90beef/%2FEvents%3B1/0-40/Muon_eta%2C%21load%2C%21content",
                "a9490124-3648-11ea-89e9-f5b55c90beef/%2FEvents%3B1/0-40/Muon_mass%2C%21load%2C%21content",
                "a9490124-3648-11ea-89e9-f5b55c90beef/%2FEvents%3B1/0-40/Muon_charge%2C%21load%2C%21content",
            ]
        ),
        executor=processor.dask_executor,
        executor_args=exe_args,
    )

    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66
    return hists["worker"]


def test_dask_job():
    distributed = pytest.importorskip("distributed", minversion="2.6.0")
    client = distributed.Client(dashboard_address=None)

    import os
    import os.path as osp

    filelist = {
        "ZJets": [osp.join(os.getcwd(), "tests/samples/nano_dy.root")],
        "Data": [osp.join(os.getcwd(), "tests/samples/nano_dimuon.root")],
    }

    do_dask_job(client, filelist)
    do_dask_job(client, filelist, compression=2)

    filelist = {
        "ZJets": {
            "treename": "Events",
            "files": [osp.join(os.getcwd(), "tests/samples/nano_dy.root")],
            "metadata": {"checkusermeta": True, "someusermeta": "hello"},
        },
        "Data": {
            "treename": "Events",
            "files": [osp.join(os.getcwd(), "tests/samples/nano_dimuon.root")],
            "metadata": {"checkusermeta": True, "someusermeta2": "world"},
        },
    }

    do_dask_job(client, filelist)

    client.close()


def test_dask_cached():
    distributed = pytest.importorskip("distributed", minversion="2.6.0")
    client = distributed.Client(dashboard_address=None)

    import os
    import os.path as osp

    filelist = {
        "ZJets": [osp.join(os.getcwd(), "tests/samples/nano_dy.root")],
        "Data": [osp.join(os.getcwd(), "tests/samples/nano_dimuon.root")],
    }

    do_dask_cached(client, filelist)
    workers1 = do_dask_cached(client, filelist, "dask-worker")
    assert len(workers1) > 0
    workers2 = do_dask_cached(client, filelist, "dask-worker")
    assert workers1 == workers2

    filelist = {
        "ZJets": {
            "treename": "Events",
            "files": [osp.join(os.getcwd(), "tests/samples/nano_dy.root")],
            "metadata": {"checkusermeta": True, "someusermeta": "hello"},
        },
        "Data": {
            "treename": "Events",
            "files": [osp.join(os.getcwd(), "tests/samples/nano_dimuon.root")],
            "metadata": {"checkusermeta": True, "someusermeta2": "world"},
        },
    }

    do_dask_cached(client, filelist)

    client.close()
