from __future__ import print_function, division
from coffea import processor

import warnings

import numpy as np

import sys
import pytest


def do_dask_job(client, filelist):
    treename='Events'
    from coffea.processor.test_items import NanoTestProcessor
    proc = NanoTestProcessor()
    
    hists = processor.run_uproot_job(filelist,
                                     treename,
                                     processor_instance=proc,
                                     executor=processor.dask_executor,
                                     executor_args={'client': client})
    
    assert( hists['cutflow']['ZJets_pt'] == 4 )
    assert( hists['cutflow']['ZJets_mass'] == 1 )
    assert( hists['cutflow']['Data_pt'] == 15 )
    assert( hists['cutflow']['Data_mass'] == 5 )
    

def test_dask_local():
    distributed = pytest.importorskip("distributed", minversion="2.6.0")
    client = distributed.Client()
    
    import os
    import os.path as osp

    filelist = {
        'ZJets': [osp.join(os.getcwd(),'tests/samples/nano_dy.root')],
        'Data' : [osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')]
    }

    do_dask_job(client, filelist)

    filelist = {
        'ZJets': {'treename': 'Events', 'files': [osp.join(os.getcwd(),'tests/samples/nano_dy.root')]},
        'Data' : {'treename': 'Events', 'files': [osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')]}
    }
        
    do_dask_job(client, filelist)

    client.close()
