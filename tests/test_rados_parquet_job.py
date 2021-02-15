import os
import sys
import uproot
import awkward as ak
from coffea import processor, hist, nanoevents
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor.test_items import NanoEventsProcessor


if __name__ == "__main__":
    ak.to_parquet(
        uproot.lazy("tests/samples/nano_dy.root:Events"),
        "nano_dy.parquet",
        list_to32=True,
        use_dictionary=False,
        compression="GZIP",
        compression_level=1,
    )

    ak.to_parquet(
        uproot.lazy("tests/samples/nano_dimuon.root:Events"),
        "nano_dimuon.parquet",
        list_to32=True,
        use_dictionary=False,
        compression="GZIP",
        compression_level=1,
    )

    os.makedirs("/mnt/cephfs/nanoevents/ZJets")
    os.makedirs("/mnt/cephfs/nanoevents/Data")
    for i in range(6):
        os.system(f"cp nano_dy.parquet /mnt/cephfs/nanoevents/ZJets/nano_dy.{i}.parquet")
        os.system(f"cp nano_dimuon.parquet /mnt/cephfs/nanoevents/Data/nano_dimuon.{i}.parquet")

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(processes=True, threads_per_worker=1)
    client = Client(cluster)

    executor_args = {
        "client": client, 
        "ceph_config_path": "/etc/ceph/ceph.conf"
    }

    hists = processor.run_rados_parquet_job({
            "ZJets": "/mnt/cephfs/nanoevents/ZJets",
            "Data": "/mnt/cephfs/nanoevents/Data"
        },
        "Events",
        processor_instance=NanoEventsProcessor(),
        executor=processor.dask_executor,
        executor_args=executor_args
    )

    assert( hists['cutflow']['ZJets_pt'] == 108 )
    assert( hists['cutflow']['ZJets_mass'] == 36 )
    assert( hists['cutflow']['Data_pt'] == 504 )
    assert( hists['cutflow']['Data_mass'] == 396 )
