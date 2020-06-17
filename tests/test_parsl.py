from __future__ import print_function, division
from coffea import processor

import multiprocessing
import sys
import pytest

if sys.version.startswith("3.8") and sys.platform.startswith("darwin"):
    pytest.skip("parsl not yet functional in python 3.8 on macs", allow_module_level=True)

def test_parsl_start_stop():
    parsl = pytest.importorskip("parsl", minversion="0.7.2")

    from coffea.processor.parsl.detail import (_parsl_initialize,
                                               _parsl_stop,
                                               _default_cfg)

    _parsl_initialize(config=_default_cfg)

    _parsl_stop()


def do_parsl_job(filelist, flatten=False, compression=0, config=None):
    treename='Events'

    from coffea.processor.test_items import NanoTestProcessor
    proc = NanoTestProcessor()

    exe_args = {
        'flatten': flatten,
        'compression': compression,
        'config': config,
    }
    hists = processor.run_parsl_job(filelist,
                                    treename,
                                    processor_instance=proc,
                                    executor=processor.executor.parsl_executor,
                                    executor_args=exe_args)

    assert( hists['cutflow']['ZJets_pt'] == 18 )
    assert( hists['cutflow']['ZJets_mass'] == 6 )
    assert( hists['cutflow']['Data_pt'] == 84 )
    assert( hists['cutflow']['Data_mass'] == 66 )


@pytest.mark.skipif(sys.platform.startswith('darwin'), reason='parsl htex not working on osx again')
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
    parsl.load(parsl_config)

    filelist = {
        'ZJets': [osp.join(os.getcwd(),'tests/samples/nano_dy.root')],
        'Data' : [osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')]
    }

    do_parsl_job(filelist)
    do_parsl_job(filelist, compression=1)

    filelist = {
        'ZJets': {'treename': 'Events', 'files': [osp.join(os.getcwd(),'tests/samples/nano_dy.root')]},
        'Data' : {'treename': 'Events', 'files': [osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')]}
    }

    do_parsl_job(filelist)
    do_parsl_job(filelist, flatten=True)


def test_parsl_deprecated():
    parsl = pytest.importorskip("parsl", minversion="0.7.2")
    
    from coffea.processor.parsl.parsl_executor import coffea_pyapp_func, parsl_executor

    with pytest.raises(RuntimeError):
        coffea_pyapp_func(None, None, None, None, None, None)

    with pytest.raises(RuntimeError):
        parsl_executor(None, None, None, None)


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
