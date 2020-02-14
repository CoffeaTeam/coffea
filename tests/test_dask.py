from __future__ import print_function, division
from coffea import processor

import warnings

import numpy as np

import sys
import pytest


def do_dask_job(client, filelist, compression=0):
    treename='Events'
    from coffea.processor.test_items import NanoTestProcessor
    proc = NanoTestProcessor()
    
    exe_args = {
        'client': client,
        'compression': compression,
    }
    hists = processor.run_uproot_job(filelist,
                                     treename,
                                     processor_instance=proc,
                                     executor=processor.dask_executor,
                                     executor_args=exe_args)
    
    assert( hists['cutflow']['ZJets_pt'] == 18 )
    assert( hists['cutflow']['ZJets_mass'] == 6 )
    assert( hists['cutflow']['Data_pt'] == 84 )
    assert( hists['cutflow']['Data_mass'] == 66 )
    

def do_dask_cached(client, filelist, cachestrategy=None):
    from coffea.processor.test_items import NanoEventsProcessor
    from coffea.processor.dask import register_columncache
    register_columncache(client)

    exe_args = {
        'client': client,
        'nano': True,
        'cachestrategy': cachestrategy,
        'savemetrics': True,
        'worker_affinity': True if cachestrategy is not None else False,
    }
    hists, metrics = processor.run_uproot_job(
        filelist,
        'Events',
        processor_instance=NanoEventsProcessor(canaries=['0001a210a3f8364811eaa29ff5b55c90beef;Events;0;40;Muon_pt']),
        executor=processor.dask_executor,
        executor_args=exe_args
    )

    assert( hists['cutflow']['ZJets_pt'] == 18 )
    assert( hists['cutflow']['ZJets_mass'] == 6 )
    assert( hists['cutflow']['Data_pt'] == 84 )
    assert( hists['cutflow']['Data_mass'] == 66 )
    return hists['worker']


def test_dask_local():
    distributed = pytest.importorskip("distributed", minversion="2.6.0")
    # `python setup.py pytest` doesn't seem to play nicely with separate processses
    client = distributed.Client(processes=False, dashboard_address=None)

    import os
    import os.path as osp

    filelist = {
        'ZJets': [osp.join(os.getcwd(),'tests/samples/nano_dy.root')],
        'Data' : [osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')]
    }

    do_dask_job(client, filelist)
    do_dask_job(client, filelist, compression=2)
    do_dask_cached(client, filelist)
    workers1 = do_dask_cached(client, filelist, 'dask-worker')
    assert len(workers1) > 0
    workers2 = do_dask_cached(client, filelist, 'dask-worker')
    assert workers1 == workers2

    filelist = {
        'ZJets': {'treename': 'Events', 'files': [osp.join(os.getcwd(),'tests/samples/nano_dy.root')]},
        'Data' : {'treename': 'Events', 'files': [osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')]}
    }

    do_dask_job(client, filelist)

    client.close()


if __name__ == '__main__':
    test_dask_local()
