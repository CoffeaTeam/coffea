import os
from sys import version_info

import hist.dask as hda
import pytest

from coffea import processor
from coffea.nanoevents import NanoEventsFactory


def histogram_common():
    # The opendata files are non-standard NanoAOD, so some optional data columns are missing
    processor.NanoAODSchema.warn_missing_crossrefs = False

    # "file:/tmp/Run2012B_SingleMu.root",
    events = NanoEventsFactory.from_root(
        {os.path.abspath("tests/samples/nano_dy.root"): "Events"},
        chunks_per_file=4,
        metadata={"dataset": "SingleMu"},
    ).events()

    q1_hist = (
        hda.Hist.new.Reg(100, 0, 200, name="met", label="$E_{T}^{miss}$ [GeV]")
        .Double()
        .fill(events.MET.pt)
    )

    return q1_hist


def test_taskvine_local_env():
    try:
        from ndcctools.taskvine import DaskVine, Factory
    except ImportError:
        print("taskvine is not installed. Omitting test.")
        return

    m = DaskVine(port=0)
    workers = Factory(manager=m, batch_type="local")
    workers.min_workers = 1
    workers.max_workers = 1
    workers.cores = 1

    q1_hist = histogram_common()
    with workers:
        result = q1_hist.compute(
            scheduler=m.get, resources={"cores": 1}, resources_mode=None
        )
        assert result.sum() == 40.0


@pytest.mark.skip(reason="need to investigate how to do conda-pack in ci")
def test_taskvine_remote_env():
    try:
        from ndcctools.poncho import package_create
        from ndcctools.taskvine import DaskVine, Factory
    except ImportError:
        print("taskvine is not installed. Omitting test.")
        return

    py_version = f"{version_info[0]}.{version_info[1]}.{version_info[2]}"

    poncho_spec = {
        "conda": {
            "channels": ["conda-forge"],
            "dependencies": [
                f"python={py_version}",
                "ndcctools",
                "pip",
            ],
        },
        "pip": ["coffea"],
    }

    env_filename = "vine-env.tar.gz"
    package_create.pack_env_from_dict(poncho_spec, env_filename)

    m = DaskVine(port=0)
    env = m.declare_poncho(env_filename, cache=True)

    workers = Factory(manager=m, batch_type="local")
    workers.min_workers = 1
    workers.max_workers = 1
    workers.cores = 1

    q1_hist = histogram_common()
    with workers:
        result = q1_hist.compute(
            scheduler=m.get,
            resources={"cores": 1},
            resources_mode=None,
            environment=env,
        )
        assert result.sum() == 40.0
