import contextlib
import pickle

import awkward as ak
import numpy as np
import pytest
import uproot

from coffea.dataset_tools import preprocess
from coffea.dataset_tools.filespec import DataGroupSpec
from coffea.util import decompress_form

pytest.importorskip("graphed")
from graphed.core.execution import SequentialRunner  # noqa: E402

N = 20


def _write_flag_files(tmp_path):
    """ROOT and parquet pairs where only the first file of each pair has a bool ``flag``."""
    has_root, missing_root = str(tmp_path / "has.root"), str(tmp_path / "missing.root")
    with uproot.recreate(has_root) as f:
        f.mktree("Events", {"x": np.float32, "flag": np.bool_})
        f["Events"].extend(
            {"x": np.arange(N, dtype=np.float32), "flag": np.ones(N, dtype=bool)}
        )
    with uproot.recreate(missing_root) as f:
        f.mktree("Events", {"x": np.float32})
        f["Events"].extend({"x": np.arange(N, dtype=np.float32) + 100})
    has_pq, missing_pq = str(tmp_path / "has.parquet"), str(
        tmp_path / "missing.parquet"
    )
    ak.to_parquet(
        ak.zip({"x": np.arange(N, dtype=np.float32), "flag": np.ones(N, dtype=bool)}),
        has_pq,
    )
    ak.to_parquet(ak.zip({"x": np.arange(N, dtype=np.float32) + 100}), missing_pq)
    return has_root, missing_root, has_pq, missing_pq


class _PicklingRunner:
    """A graphed executor that ships the plan through pickle, as a process pool does."""

    def __init__(self):
        self.n_tasks = 0

    def run(self, plan):
        self.n_tasks += len(plan.tasks)
        return SequentialRunner().run(pickle.loads(pickle.dumps(plan)))


def _root_fileset(tmp_path):
    has_root, missing_root, _, _ = _write_flag_files(tmp_path)
    bad = str(tmp_path / "does_not_exist.root")
    return {
        "union": {"files": {has_root: "Events", missing_root: "Events", bad: "Events"}},
        "single": {"files": {has_root: "Events"}},
    }


@pytest.mark.parametrize("files_per_batch", [1, 2])
@pytest.mark.parametrize("save_form", [False, True])
@pytest.mark.parametrize("fileset_kind", ["legacy", "dict", "mixed"])
def test_preprocess_graphed_matches_dask(
    tmp_path, fileset_kind, save_form, files_per_batch
):
    pytest.importorskip("dask_awkward")
    fileset = _root_fileset(tmp_path)
    kwargs = dict(
        step_size=7,
        files_per_batch=files_per_batch,
        skip_bad_files=True,
        save_form=save_form,
        preprocess_legacy_root=fileset_kind == "legacy",
    )
    if fileset_kind == "mixed":
        _, _, has_pq, missing_pq = _write_flag_files(tmp_path)
        fileset["parquet"] = {"files": {has_pq: None, missing_pq: None}}
        fileset = DataGroupSpec(fileset)

    with (
        pytest.warns(DeprecationWarning)
        if fileset_kind == "dict"
        else contextlib.nullcontext()
    ):
        dask_out = preprocess(fileset, scheduler="sync", **kwargs)
    with (
        pytest.warns(DeprecationWarning)
        if fileset_kind == "dict"
        else contextlib.nullcontext()
    ):
        runner = _PicklingRunner()
        graphed_out = preprocess(fileset, scheduler=runner, mode="graphed", **kwargs)

    assert graphed_out == dask_out
    # one task per batch: union has 3 files, single 1, parquet 2
    n_files = [3, 1] + ([2] if fileset_kind == "mixed" else [])
    assert runner.n_tasks == sum(-(-n // files_per_batch) for n in n_files)

    available = graphed_out[0]
    union = available["union"]
    assert len(union["files"] if isinstance(union, dict) else union.files) == 2
    if save_form:
        datasets = ["union"] + (["parquet"] if fileset_kind == "mixed" else [])
        for name in datasets:
            ds = available[name]
            if isinstance(ds, dict):
                form = decompress_form(ds.get("form") or ds["compressed_form"])
            else:
                form = ds.form
            assert "IndexedOption" in str(form)


def test_preprocess_graphed_refuses_dask_scheduler(tmp_path):
    has_root, *_ = _write_flag_files(tmp_path)
    with pytest.raises(TypeError, match="graphed executor"):
        preprocess(
            DataGroupSpec({"ds": {"files": {has_root: "Events"}}}),
            scheduler="sync",
            mode="graphed",
        )
