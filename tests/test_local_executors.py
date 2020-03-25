import sys
import os
import os.path as osp

import pytest

from coffea import hist, processor

if sys.platform.startswith("win"):
    pytest.skip("skipping tests that only function in linux", allow_module_level=True)


def template_analysis(filelist, executor, flatten, compression, align_clusters=False):
    from coffea.processor import run_uproot_job
    
    treename='Events'

    from coffea.processor.test_items import NanoTestProcessor

    exe_args = {
        'workers': 1,
        'pre_workers': 1,
        'flatten': flatten,
        'compression': compression,
        'align_clusters': align_clusters,
    }

    hists = run_uproot_job(filelist, treename, NanoTestProcessor(), executor,
                           executor_args = exe_args)

    print(hists)
    assert( hists['cutflow']['ZJets_pt'] == 18 )
    assert( hists['cutflow']['ZJets_mass'] == 6 )
    assert( hists['cutflow']['Data_pt'] == 84 )
    assert( hists['cutflow']['Data_mass'] == 66 )


def test_iterative_executor():
    from coffea.processor import iterative_executor
    
    filelist = {
        'ZJets': [osp.abspath('tests/samples/nano_dy.root')],
        'Data': [osp.abspath('tests/samples/nano_dimuon.root')]
    }
    
    template_analysis(filelist, iterative_executor, flatten=True, compression=0)
    template_analysis(filelist, iterative_executor, flatten=True, compression=1)
    template_analysis(filelist, iterative_executor, flatten=False, compression=2)
    template_analysis(filelist, iterative_executor, flatten=False, compression=1, align_clusters=True)

    filelist = {
        'ZJets': {'treename': 'Events', 'files': [osp.abspath('tests/samples/nano_dy.root')]},
        'Data': {'treename': 'Events', 'files': [osp.abspath('tests/samples/nano_dimuon.root')]}
        }

    template_analysis(filelist, iterative_executor, flatten=True, compression=0)


def test_futures_executor():

    if sys.version.startswith("3.8") and sys.platform.startswith("darwin"):
        pytest.skip("futures not yet functional in python 3.8 on macs")

    from coffea.processor import futures_executor
    
    filelist = {
        'ZJets': [osp.abspath('tests/samples/nano_dy.root')],
        'Data': [osp.abspath('tests/samples/nano_dimuon.root')]
    }

    template_analysis(filelist, futures_executor, flatten=True, compression=0)
    template_analysis(filelist, futures_executor, flatten=True, compression=1, align_clusters=True)
    template_analysis(filelist, futures_executor, flatten=False, compression=0)
    template_analysis(filelist, futures_executor, flatten=False, compression=1)

    filelist = {
        'ZJets': {'treename': 'Events', 'files': [osp.abspath('tests/samples/nano_dy.root')]},
        'Data': {'treename':'Events', 'files': [osp.abspath('tests/samples/nano_dimuon.root')]}
    }
    
    template_analysis(filelist, futures_executor, flatten=True, compression=2)
    template_analysis(filelist, futures_executor, flatten=False, compression=0)
