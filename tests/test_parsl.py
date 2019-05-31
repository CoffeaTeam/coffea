from __future__ import print_function, division
from fnal_column_analysis_tools import processor

import multiprocessing

import warnings

import numpy as np

def test_parsl_start_stop():
    try:
        import parsl
    except ImportError:
        warnings.warn('parsl not installed, skipping tests')
        return
    except Exception as e:
        warnings.warn('other error when trying to import parsl!')
        raise e

    from fnal_column_analysis_tools.processor.parsl.detail import (_parsl_initialize,
                                                                   _parsl_stop,
                                                                   _default_cfg)

    dfk = _parsl_initialize(config=_default_cfg)
    
    _parsl_stop(dfk)

def test_parsl_executor():
    try:
        import parsl
    except ImportError:
        warnings.warn('parsl not installed, skipping tests')
        return
    except Exception as e:
        warnings.warn('other error when trying to import parsl!')
        raise e
    
    from fnal_column_analysis_tools.processor import run_parsl_job

    from fnal_column_analysis_tools.processor.parsl.detail import (_parsl_initialize,
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

    from fnal_column_analysis_tools.processor.test_items import NanoTestProcessor
    from fnal_column_analysis_tools.processor.parsl.parsl_executor import parsl_executor

    dfk = _parsl_initialize(parsl_config)

    proc = NanoTestProcessor()

    hists = run_parsl_job(filelist, treename, processor_instance = proc, executor=parsl_executor, data_flow=dfk)

    _parsl_stop(dfk)

    assert( hists['cutflow']['ZJets_pt'] == 4 )
    assert( hists['cutflow']['ZJets_mass'] == 1 )
    assert( hists['cutflow']['Data_pt'] == 15 )
    assert( hists['cutflow']['Data_mass'] == 5 )
