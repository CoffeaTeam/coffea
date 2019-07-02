from __future__ import print_function, division
from coffea import processor

import multiprocessing

import warnings

import numpy as np

import sys
import pytest


def test_parsl_start_stop():
    parsl = pytest.importorskip("parsl", minversion="0.7.2")
    
    from coffea.processor.parsl.detail import (_parsl_initialize,
                                               _parsl_stop,
                                               _default_cfg)

    dfk = _parsl_initialize(config=_default_cfg)
    
    _parsl_stop(dfk)


def do_parsl_job(parsl_config, filelist):
    from coffea.processor.parsl.detail import (_parsl_initialize,
                                               _parsl_stop)
    from coffea.processor import run_parsl_job

    import os
    import os.path as osp
    
    treename='Events'

    from coffea.processor.test_items import NanoTestProcessor
    from coffea.processor.parsl.parsl_executor import parsl_executor
    
    dfk = _parsl_initialize(parsl_config)
    
    proc = NanoTestProcessor()
    
    hists = run_parsl_job(filelist, treename, processor_instance = proc, executor=parsl_executor, data_flow=dfk)
    
    hists2 = run_parsl_job(filelist, treename, processor_instance = proc, executor=parsl_executor, data_flow=dfk, executor_args={'flatten': False})
    
    _parsl_stop(dfk)
    
    assert( hists['cutflow']['ZJets_pt'] == 4 )
    assert( hists['cutflow']['ZJets_mass'] == 1 )
    assert( hists['cutflow']['Data_pt'] == 15 )
    assert( hists['cutflow']['Data_mass'] == 5 )
    
    assert( hists2['cutflow']['ZJets_pt'] == 4 )
    assert( hists2['cutflow']['ZJets_mass'] == 1 )
    assert( hists2['cutflow']['Data_pt'] == 15 )
    assert( hists2['cutflow']['Data_mass'] == 5 )


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason='issue with parsl htex on macos')
def test_parsl_htex_executor():
    parsl = pytest.importorskip("parsl", minversion="0.7.2")
    import os
    import os.path as osp

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
                cores_per_worker=max(multiprocessing.cpu_count()//2, 1),
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

    filelist = {
        'ZJets': [osp.join(os.getcwd(),'tests/samples/nano_dy.root')],
        'Data' : [osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')]
    }

    do_parsl_job(parsl_config, filelist)

    parsl_config = Config(
        executors=[
            HighThroughputExecutor(
                label="coffea_parsl_default",
                address=address_by_hostname(),
                cores_per_worker=max(multiprocessing.cpu_count()//2, 1),
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

    filelist = {
        'ZJets': {'treename': 'Events', 'files': [osp.join(os.getcwd(),'tests/samples/nano_dy.root')]},
        'Data' : {'treename': 'Events', 'files': [osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')]}
    }
        
    do_parsl_job(parsl_config, filelist)


def test_parsl_funcs():
    parsl = pytest.importorskip("parsl", minversion="0.7.2")

    import os.path as osp
    from coffea.processor.parsl.detail import derive_chunks

    filename = osp.abspath('tests/samples/nano_dy.root')
    dataset = 'Z+Jets'
    treename = 'Events'
    chunksize = 20
    ds, tn, test = derive_chunks.func(filename, treename, chunksize, dataset)
    
    assert(dataset == ds)
    assert(treename == tn)
    assert('nano_dy.root' in test[0][0])
    assert(test[0][1] == 20)
    assert(test[0][2] == 0)

    from coffea.processor.parsl.parsl_executor import coffea_pyapp
    from coffea.processor.test_items import NanoTestProcessor
    import pickle as pkl
    import cloudpickle as cpkl
    import lz4.frame as lz4f
    
    procpkl = lz4f.compress(cpkl.dumps(NanoTestProcessor()))
    
    out = coffea_pyapp.func('ZJets', filename, treename, chunksize, 0, procpkl)

    hists = pkl.loads(lz4f.decompress(out[0]))
    assert( hists['cutflow']['ZJets_pt'] == 4 )
    assert( hists['cutflow']['ZJets_mass'] == 1 )
    assert(out[1] == 10)
    assert(out[2] == 'ZJets')

@pytest.mark.skipif(sys.platform.startswith('win'), reason='signals are different on windows')
def test_timeout():
    from coffea.processor.parsl.timeout import timeout
    import signal
    
    @timeout
    def too_long(timeout=None):
        import time
        time.sleep(20)

    @timeout
    def make_except(timeout=None):
        import time
        time.sleep(1)
        raise Exception('oops!')

    try:
        too_long(timeout=5)
    except Exception as e:
        assert(e.args[0] == "Timeout hit")

    try:
        make_except(timeout=20)
    except Exception as e:
        assert(e.args[0] == 'oops!')

    # reset alarms for other tests, this is suspicious
    signal.alarm(0)


def test_parsl_condor_cfg():
    parsl = pytest.importorskip("parsl", minversion="0.7.2")

    from coffea.processor.parsl.condor_config import condor_config

    test = condor_config()


def test_parsl_slurm_cfg():
    import os
    parsl = pytest.importorskip("parsl", minversion="0.7.2")

    x509_proxy = 'x509up_u%s' % (os.getuid())
    fname = '/tmp/%s' % x509_proxy
    with open(fname, 'w+'):
        os.utime(fname, None)
    
    from coffea.processor.parsl.slurm_config import slurm_config

    test = slurm_config()
