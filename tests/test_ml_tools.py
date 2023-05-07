import awkward as ak
import dask_awkward as dak
import numpy as np
import pytest


def prepare_jets_array(njets):
    # Creating jagged Jet-with-constituent array, returning both awkward and lazy
    # dask_awkward arrays
    NFEAT = 100
    jets = ak.zip(
        {
            "pt": ak.from_numpy(np.random.random(size=njets)),
            "eta": ak.from_numpy(np.random.random(size=njets)),
            "phi": ak.from_numpy(np.random.random(size=njets)),
            "ncands": ak.from_numpy(np.random.randint(1, 50, size=njets)),
        },
        with_name="LorentzVector",
    )
    pfcands = ak.zip(
        {
            "pt": ak.from_regular(np.random.random(size=(njets, NFEAT))),
            "eta": ak.from_regular(np.random.random(size=(njets, NFEAT))),
            "phi": ak.from_regular(np.random.random(size=(njets, NFEAT))),
            "feat1": ak.from_regular(np.random.random(size=(njets, NFEAT))),
            "feat2": ak.from_regular(np.random.random(size=(njets, NFEAT))),
        },
        with_name="LorentzVector",
    )

    idx = ak.local_index(pfcands.pt, axis=-1)
    pfcands = pfcands[idx < jets.ncands]
    jets["pfcands"] = pfcands[:]

    ak_jets = jets[:]
    ak.to_parquet(jets, "ml_tools.parquet")
    dak_jets = dak.from_parquet("ml_tools.parquet")
    return ak_jets, dak_jets


def common_awkward_to_numpy(jets):
    def my_pad(arr):
        return ak.fill_none(ak.pad_none(arr, 100, axis=1, clip=True), 0.0)

    fmap = {
        "points": {
            "deta": my_pad(jets.eta - jets.pfcands.eta),
            "dphi": my_pad(jets.phi - jets.pfcands.phi),
        },
        "features": {
            "dr": my_pad(
                np.sqrt(
                    (jets.eta - jets.pfcands.eta) ** 2
                    + (jets.phi - jets.pfcands.phi) ** 2
                )
            ),
            "lpt": my_pad(np.log(jets.pfcands.pt)),
            "lptf": my_pad(np.log(jets.pfcands.pt / ak.sum(jets.pfcands.pt, axis=-1))),
            "f1": my_pad(np.log(jets.pfcands.feat1 + 1)),
            "f2": my_pad(np.log(jets.pfcands.feat2 + 1)),
        },
        "mask": {
            "mask": my_pad(ak.ones_like(jets.pfcands.pt)),
        },
    }

    return {
        k: ak.concatenate(
            [x[:, np.newaxis, :] for x in fmap[k].values()], axis=1
        ).to_numpy()
        for k in fmap.keys()
    }


def test_triton():
    _ = pytest.importorskip("tritonclient")    

    from coffea.ml_tools.triton_wrapper import triton_wrapper

    # Defining custom wrapper function with awkward padding requirements.
    class triton_wrapper_test(triton_wrapper):
        def awkward_to_numpy(self, output_list, jets):
            return [], {
                "output_list": output_list,
                "input_dict": common_awkward_to_numpy(jets),
            }

        def dask_columns(self, output_list, jets):
            return [
                jets.eta,
                jets.phi,
                jets.pfcands.pt,
                jets.pfcands.phi,
                jets.pfcands.eta,
                jets.pfcands.feat1,
                jets.pfcands.feat2,
            ]

    # Running the evaluation in lazy and non-lazy forms
    tw = triton_wrapper_test(
        model_url="triton+grpc://localhost:8001/pn_test/1",
        client_args=dict(
            ssl=False
        ),  # Solves SSL version mismatch for local inference server
    )

    ak_jets, dak_jets = prepare_jets_array(njets=256)

    # Vanilla awkward arrays
    ak_res = tw(["output"], ak_jets)
    dak_res = tw(["output"], dak_jets)

    for k in ak_res.keys():
        assert ak.all(ak_res[k] == dak_res[k].compute())
    columns = list(dak.necessary_columns(dak_res).values())[0]
    assert sorted(columns) == sorted(
        [
            "eta",
            "phi",
            "pfcands.pt",
            "pfcands.phi",
            "pfcands.eta",
            "pfcands.feat1",
            "pfcands.feat2",
        ]
    )


def test_torch():
    torch = pytest.importorskip("torch")

    from coffea.ml_tools.torch_wrapper import torch_wrapper

    class torch_wrapper_test(torch_wrapper):
        def awkward_to_numpy(self, jets):
            default = common_awkward_to_numpy(jets)
            return [], {
                "points": default["points"].astype(np.float32),
                "features": default["features"].astype(np.float32),
                "mask": default["mask"].astype(np.float16),
            }

        def dask_columns(self, jets):
            return [
                jets.eta,
                jets.phi,
                jets.pfcands.pt,
                jets.pfcands.phi,
                jets.pfcands.eta,
                jets.pfcands.feat1,
                jets.pfcands.feat2,
            ]

    model = torch.jit.load("tests/samples/pn_demo.pt")
    tw = torch_wrapper_test(model)
    ak_jets, dak_jets = prepare_jets_array(njets=256)

    ak_res = tw(ak_jets)
    dak_res = tw(dak_jets)

    assert np.all(np.isclose(ak_res, dak_res.compute()))


def test_xgboost():
    _ = pytest.importorskip("xgboost")

    from coffea.ml_tools.xgboost_wrapper import xgboost_wrapper

    feature_list = [f"feat{i}" for i in range(10)]

    class xgboost_test(xgboost_wrapper):
        def awkward_to_numpy(self, events):
            ret = np.column_stack([events[name].to_numpy() for name in feature_list])
            return [], dict(data=ret)

        def dask_columns(self, events):
            return [events[f] for f in feature_list]

    xgb_wrap = xgboost_test("tests/samples/xgboost_example.xgb")

    # Dummy 1000 event array with 20 feature branches
    ak_events = ak.zip(
        {f"feat{i}": ak.from_numpy(np.random.random(size=1_000)) for i in range(20)}
    )
    ak.to_parquet(ak_events, "ml_tools.xgboost.parquet")
    dak_events = dak.from_parquet("ml_tools.xgboost.parquet")

    ak_res = xgb_wrap(ak_events)
    dak_res = xgb_wrap(dak_events)

    # Results should be identical
    assert ak.all(ak_res == dak_res.compute())

    # Should only load required columns
    columns = list(dak.necessary_columns(dak_res).values())[0]
    assert sorted(columns) == sorted(feature_list)
