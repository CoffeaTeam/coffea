from __future__ import print_function, division
from coffea import processor

import multiprocessing

import warnings

import numpy as np

import pytest



def test_parsl_start_stop():
    parsl = pytest.importorskip("parsl", minversion="0.7.2")
    
    from coffea.processor.parsl.detail import (_parsl_initialize,
                                               _parsl_stop,
                                               _default_cfg)

    dfk = _parsl_initialize(config=_default_cfg)
    
    _parsl_stop(dfk)


def test_parsl_executor():
    parsl = pytest.importorskip("parsl", minversion="0.7.2")
    
    from coffea.processor import run_parsl_job

    from coffea.processor.parsl.detail import (_parsl_initialize,
                                               _parsl_stop)

    from parsl.providers import LocalProvider
    from parsl.channels import LocalChannel
    from parsl.executors import HighThroughputExecutor
    from parsl.addresses import address_by_hostname
    from parsl.config import Config
    parsl_config = Config(
        executors=[
                   HighThroughputExecutor(
                    label="coffea_parsl_default",
                    address=address_by_hostname(),
                    cores_per_worker=max(multiprocessing.cpu_count()//2,1),
                    max_workers=1,
                    provider=LocalProvider(
                        channel=LocalChannel(),
                        init_blocks=1,
                        max_blocks=1,
                        nodes_per_block=1
                        ),
                    )
                   ],
        strategy=None,
        )

    import os
    import os.path as osp

    filelist = {'ZJets': [osp.join(os.getcwd(),'tests/samples/nano_dy.root')],
                'Data'  : [osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')]
        }
    treename='Events'

    from coffea.processor.test_items import NanoTestProcessor
    from coffea.processor.parsl.parsl_executor import parsl_executor

    dfk = _parsl_initialize(parsl_config)

    proc = NanoTestProcessor()

    hists = run_parsl_job(filelist, treename, processor_instance = proc, executor=parsl_executor, data_flow=dfk)

    _parsl_stop(dfk)

    assert( hists['cutflow']['ZJets_pt'] == 4 )
    assert( hists['cutflow']['ZJets_mass'] == 1 )
    assert( hists['cutflow']['Data_pt'] == 15 )
    assert( hists['cutflow']['Data_mass'] == 5 )
