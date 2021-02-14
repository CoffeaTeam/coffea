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

    os.makedirs("/mnt/cephfs/nanoevents/")
    for i in range(6):
        os.system(f"cp nano_dy.parquet /mnt/cephfs/nanoevents/nano_dy.{i}.parquet")

    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(processes=True, threads_per_worker=1)
    client = Client(cluster)

    executor_args = {
        "client": client, 
        "ceph_config_path": "/etc/ceph/ceph.conf"
    }

    hists = processor.run_rados_parquet_job(
        '/mnt/cephfs/nanoevents/',
        "Events",
        processor_instance=NanoEventsProcessor(),
        executor=processor.dask_executor,
        executor_args=executor_args
    )

    assert( hists['cutflow']['dataset_pt'] == 108 )
    assert( hists['cutflow']['dataset_mass'] == 36 )
