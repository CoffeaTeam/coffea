import pytest

from coffea import processor


def do_dask_job(client, filelist, compression=0):
    from coffea.processor.test_items import NanoTestProcessor

    executor = processor.DaskExecutor(client=client, compression=compression)
    run = processor.Runner(executor=executor)

    hists = run(filelist, "Events", processor_instance=NanoTestProcessor())

    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66


def do_dask_cached(client, filelist, cachestrategy=None):
    from coffea.nanoevents import schemas
    from coffea.processor.dask import register_columncache
    from coffea.processor.test_items import NanoEventsProcessor

    register_columncache(client)

    worker_affinity = True if cachestrategy is not None else False
    executor = processor.DaskExecutor(client=client, worker_affinity=worker_affinity)
    run = processor.Runner(
        executor=executor,
        schema=schemas.NanoAODSchema,
        cachestrategy=cachestrategy,
        savemetrics=True,
    )

    hists, metrics = run(
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

    client.wait_for_workers(1)

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


@pytest.mark.skip(reason="worker not showing up with latest dask")
def test_dask_cached():
    distributed = pytest.importorskip("distributed", minversion="2.6.0")
    client = distributed.Client(dashboard_address=None)

    import os
    import os.path as osp

    filelist = {
        "ZJets": [osp.join(os.getcwd(), "tests/samples/nano_dy.root")],
        "Data": [osp.join(os.getcwd(), "tests/samples/nano_dimuon.root")],
    }

    client.wait_for_workers(1)

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
