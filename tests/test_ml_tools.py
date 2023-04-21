import awkward as ak
import dask_awkward as dak
import numpy as np

import coffea.ml_tools

if __name__ == "__main__":
    ## Creating jagged Jet-with-constituent array
    NJETS = 2000
    NFEAT = 100
    jets = ak.zip(
        {
            "pt": ak.from_numpy(np.random.random(size=NJETS)),
            "eta": ak.from_numpy(np.random.random(size=NJETS)),
            "phi": ak.from_numpy(np.random.random(size=NJETS)),
            "ntrk": ak.from_numpy(np.random.randint(1, 50, size=NJETS)),
        }
    )
    tracks = ak.zip(
        {
            "pt": ak.from_regular(np.random.random(size=(NJETS, NFEAT))),
            "eta": ak.from_regular(np.random.random(size=(NJETS, NFEAT))),
            "phi": ak.from_regular(np.random.random(size=(NJETS, NFEAT))),
            "ip2d": ak.from_regular(np.random.random(size=(NJETS, NFEAT))),
            "ipz": ak.from_regular(np.random.random(size=(NJETS, NFEAT))),
        }
    )

    idx = ak.local_index(tracks.pt, axis=-1)
    tracks = tracks[idx < jets.ntrk]
    jets["tracks"] = tracks[:]

    # Defining custom wrapper function with awkward padding requirements.
    class triton_wrapper_test(coffea.ml_tools.triton_wrapper):
        def awkward_to_numpy(self, jets):
            def my_pad(arr):
                return ak.fill_none(ak.pad_none(arr, NFEAT, axis=1, clip=True), 0.0)

            fmap = {
                "points__0": {
                    "deta": my_pad(jets.eta - jets.tracks.eta),
                    "dphi": my_pad(jets.phi - jets.tracks.phi),
                },
                "features__1": {
                    "dr": my_pad(
                        np.sqrt(
                            (jets.eta - jets.tracks.eta) ** 2
                            + (jets.phi - jets.tracks.phi) ** 2
                        )
                    ),
                    "lpt": my_pad(np.log(jets.tracks.pt)),
                    "lptf": my_pad(
                        np.log(jets.tracks.pt / ak.sum(jets.tracks.pt, axis=-1))
                    ),
                    "ip2d": my_pad(
                        np.sign(jets.tracks.ip2d) * np.log(jets.tracks.ip2d + 1)
                    ),
                    "ipz": my_pad(
                        np.sign(jets.tracks.ipz) * np.log(jets.tracks.ipz + 1)
                    ),
                },
                "mask__2": {
                    "mask": my_pad(ak.ones_like(jets.tracks.pt)),
                },
            }

            return {
                k: ak.concatenate(
                    [x[:, np.newaxis, :] for x in fmap[k].values()], axis=1
                ).to_numpy()
                for k in fmap.keys()
            }

        def dask_touch(self, jets):
            jets.eta.layout._touch_data(recursive=False)
            jets.phi.layout._touch_data(recursive=False)
            jets.tracks.pt.layout._touch_data(recursive=False)
            jets.tracks.eta.layout._touch_data(recursive=False)
            jets.tracks.ip2d.layout._touch_data(recursive=False)
            jets.tracks.ipz.layout._touch_data(recursive=False)
            pass

    # Running the evaluation in lazy and non-lazy forms
    tw = triton_wrapper_test(
        model_url="triton+grpc://triton.apps.okddev.fnal.gov:443/emj_gnn_aligned/1"
    )

    # Numpy arrays testing
    np_res = tw._numpy_call(["softmax__0"], tw.awkward_to_numpy(jets), validate=True)
    print({k: v.shape for k, v in np_res.items()})

    # Vanilla awkward arrays
    ak_res = tw(["softmax__0"], jets)
    print({k: v.to_numpy().shape for k, v in ak_res.items()})

    for k in np_res.keys():
        assert np.all(ak.to_numpy(ak_res[k]) == np_res[k])

    # DASK awkward arrays (to_parquet for force everything to be lazy)
    ak.to_parquet(jets, "triton_test.parquet")
    dak_jets = dak.from_parquet("triton_test.parquet")
    dak_res = tw(["softmax__0"], dak_jets)
    print({k: v.compute().to_numpy().shape for k, v in dak_res.items()})

    print(dak.necessary_columns(dak_res))

    for k in ak_res.keys():
        assert ak.all(ak_res[k] == dak_res[k].compute())
