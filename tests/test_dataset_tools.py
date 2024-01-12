import dask
import pytest
import uproot
from distributed import Client

from coffea.dataset_tools import (
    apply_to_fileset,
    filter_files,
    get_failed_steps_for_fileset,
    max_chunks,
    max_files,
    preprocess,
    slice_chunks,
    slice_files,
)
from coffea.nanoevents import BaseSchema, NanoAODSchema
from coffea.processor.test_items import NanoEventsProcessor, NanoTestProcessor
from coffea.util import decompress_form

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
        },
        "metadata": None,
        "form": None,
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
        },
        "metadata": None,
        "form": None,
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
        },
        "metadata": None,
        "form": None,
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
        },
        "metadata": None,
        "form": None,
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


def test_apply_to_fileset_hinted_form():
    with Client() as _:
        dataset_runnable, dataset_updated = preprocess(
            _starting_fileset,
            maybe_step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=True,
        )

        to_compute = apply_to_fileset(
            NanoEventsProcessor(),
            dataset_runnable,
            schemaclass=NanoAODSchema,
        )
        out = dask.compute(to_compute)[0]

        assert out["ZJets"]["cutflow"]["ZJets_pt"] == 18
        assert out["ZJets"]["cutflow"]["ZJets_mass"] == 6
        assert out["Data"]["cutflow"]["Data_pt"] == 84
        assert out["Data"]["cutflow"]["Data_mass"] == 66


def test_preprocess():
    with Client() as _:
        dataset_runnable, dataset_updated = preprocess(
            _starting_fileset,
            maybe_step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=True,
        )

        assert dataset_runnable == _runnable_result
        assert dataset_updated == _updated_result


def test_preprocess_calculate_form():
    with Client() as _:
        starting_fileset = _starting_fileset

        dataset_runnable, dataset_updated = preprocess(
            starting_fileset,
            maybe_step_size=7,
            align_clusters=False,
            files_per_batch=10,
            skip_bad_files=True,
            save_form=True,
        )

        raw_form_dy = uproot.dask(
            "tests/samples/nano_dy.root:Events", open_files=False, ak_add_doc=True
        ).layout.form.to_json()
        raw_form_data = uproot.dask(
            "tests/samples/nano_dimuon.root:Events", open_files=False, ak_add_doc=True
        ).layout.form.to_json()

        assert decompress_form(dataset_runnable["ZJets"]["form"]) == raw_form_dy
        assert decompress_form(dataset_runnable["Data"]["form"]) == raw_form_data


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


def test_filter_files():
    filtered_files = filter_files(_updated_result)

    assert filtered_files == {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                    "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                    "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
    }


def test_max_files():
    maxed_files = max_files(_updated_result, 1)

    assert maxed_files == {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                    "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21], [21, 28], [28, 35], [35, 40]],
                    "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
    }


def test_slice_files():
    sliced_files = slice_files(_updated_result, slice(1, None, 2))

    assert sliced_files == {
        "ZJets": {"files": {}, "metadata": None, "form": None},
        "Data": {
            "files": {
                "tests/samples/nano_dimuon_not_there.root": {
                    "object_path": "Events",
                    "steps": None,
                    "uuid": None,
                }
            },
            "metadata": None,
            "form": None,
        },
    }


def test_max_chunks():
    max_chunked = max_chunks(_runnable_result, 3)

    assert max_chunked == {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21]],
                    "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [7, 14], [14, 21]],
                    "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
    }


def test_slice_chunks():
    slice_chunked = slice_chunks(_runnable_result, slice(None, None, 2))

    assert slice_chunked == {
        "ZJets": {
            "files": {
                "tests/samples/nano_dy.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [14, 21], [28, 35]],
                    "uuid": "a9490124-3648-11ea-89e9-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
        },
        "Data": {
            "files": {
                "tests/samples/nano_dimuon.root": {
                    "object_path": "Events",
                    "steps": [[0, 7], [14, 21], [28, 35]],
                    "uuid": "a210a3f8-3648-11ea-a29f-f5b55c90beef",
                }
            },
            "metadata": None,
            "form": None,
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
