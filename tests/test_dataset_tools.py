import dask
import pytest
from distributed import Client

from coffea.dataset_tools import (
    apply_to_fileset,
    get_failed_steps_for_fileset,
    max_chunks,
    preprocess,
    slice_chunks,
)
from coffea.nanoevents import BaseSchema, NanoAODSchema
from coffea.processor.test_items import NanoEventsProcessor, NanoTestProcessor

_starting_fileset = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 5],
                    [5, 10],
                    [10, 15],
                    [15, 20],
                    [20, 25],
                    [25, 30],
                    [30, 35],
                    [35, 40],
                ],
            }
        }
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": "Events",
            "tests/samples/nano_dimuon_not_there.root": "Events",
        }
    },
}

_starting_fileset_with_steps = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 5],
                    [5, 10],
                    [10, 15],
                    [15, 20],
                    [20, 25],
                    [25, 30],
                    [30, 35],
                    [35, 40],
                ],
            }
        }
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": {
                "object_path": "Events",
                "steps": [
                    [0, 5],
                    [5, 10],
                    [10, 15],
                    [15, 20],
                    [20, 25],
                    [25, 30],
                    [30, 35],
                    [35, 40],
                ],
            },
            "tests/samples/nano_dimuon_not_there.root": {
                "object_path": "Events",
                "steps": [
                    [0, 5],
                    [5, 10],
                    [10, 15],
                    [15, 20],
                    [20, 25],
                    [25, 30],
                    [30, 35],
                    [35, 40],
                ],
            },
        }
    },
}

_runnable_result = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            }
        }
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
            }
        }
    },
}

_updated_result = {
    "ZJets": {
        "files": {
            "tests/samples/nano_dy.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
            }
        }
    },
    "Data": {
        "files": {
            "tests/samples/nano_dimuon.root": {
                "object_path": "Events",
                "steps": [
                    [0, 7],
                    [7, 14],
                    [14, 21],
                    [21, 28],
                    [28, 35],
                    [35, 40],
                ],
                "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
            },
            "tests/samples/nano_dimuon_not_there.root": {
                "object_path": "Events",
                "steps": None,
                "uuid": None,
            },
        }
    },
}


@pytest.mark.parametrize(
    "proc_and_schema",
    [(NanoTestProcessor, BaseSchema), (NanoEventsProcessor, NanoAODSchema)],
)
def test_apply_to_fileset(proc_and_schema):
    proc, schemaclass = proc_and_schema

    with Client() as _:
        to_compute = apply_to_fileset(
            proc(),
            _runnable_result,
            schemaclass=schemaclass,
        )
        out = dask.compute(to_compute)[0]

        assert out["ZJets"]["cutflow"]["ZJets_pt"] == 18
        assert out["ZJets"]["cutflow"]["ZJets_mass"] == 6
        assert out["Data"]["cutflow"]["Data_pt"] == 84
        assert out["Data"]["cutflow"]["Data_mass"] == 66

        to_compute = apply_to_fileset(
            proc(),
            max_chunks(_runnable_result, 1),
            schemaclass=schemaclass,
        )
        out = dask.compute(to_compute)[0]

        assert out["ZJets"]["cutflow"]["ZJets_pt"] == 5
        assert out["ZJets"]["cutflow"]["ZJets_mass"] == 2
        assert out["Data"]["cutflow"]["Data_pt"] == 17
        assert out["Data"]["cutflow"]["Data_mass"] == 14


def test_preprocess():
    with Client() as _:
        starting_fileset = _starting_fileset

        dataset_runnable, dataset_updated = preprocess(
            starting_fileset,
            maybe_step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=True,
        )

        assert dataset_runnable == _runnable_result
        assert dataset_updated == _updated_result


def test_preprocess_failed_file():
    with Client() as _, pytest.raises(FileNotFoundError):
        starting_fileset = _starting_fileset

        dataset_runnable, dataset_updated = preprocess(
            starting_fileset,
            maybe_step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=False,
        )


def test_maxchunks():
    max_chunked = max_chunks(_runnable_result, 3)

    assert max_chunked == {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21]],
                    "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
                }
            }
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21]],
                    "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
                }
            }
        },
    }


def test_slicechunks():
    slice_chunked = slice_chunks(_runnable_result, slice(None, None, 2))

    assert slice_chunked == {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [14, 21], [28, 35]],
                    "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
                }
            }
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [14, 21], [28, 35]],
                    "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
                }
            }
        },
    }


def test_recover_failed_chunks():
    with Client() as _:
        to_compute = apply_to_fileset(
            NanoEventsProcessor(),
            _starting_fileset_with_steps,
            schemaclass=NanoAODSchema,
            uproot_options={"allow_read_errors_with_report": True},
        )
        out, reports = dask.compute(*to_compute)

    failed_fset = get_failed_steps_for_fileset(_starting_fileset_with_steps, reports)
    assert failed_fset == {
        "Data": {
            "files": {
                "tests/samples/nano_dimuon_not_there.root": {
                    "object_path": "Events",
                    "steps": [
                        [0, 5],
                        [5, 10],
                        [10, 15],
                        [15, 20],
                        [20, 25],
                        [25, 30],
                        [30, 35],
                        [35, 40],
                    ],
                }
            }
        }
    }
