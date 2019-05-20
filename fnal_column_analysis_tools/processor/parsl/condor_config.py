import parsl
import os
from parsl.app.app import python_app, bash_app
from parsl.configs.local_threads import config

from parsl.providers import CondorProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

from parsl.addresses import address_by_hostname

x509_proxy = 'x509up_u%s' % (os.getuid(), )


def condor_config(cores_per_job=4, mem_per_core=2048,
                  pyenv_dir='%s/.local' % (os.environ['HOME'], ),
                  grid_proxy_dir='/tmp/%s' % (x509_proxy, )):
    pyenv_relpath = pyenv_dir.split('/')[-1]

    wrk_init = '''
    source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc7-opt/setup.sh
    export PATH=`pwd`/%s:$PATH
    export PYTHONPATH=`pwd`/%s:$PYTHONPATH

    export X509_USER_PROXY=`pwd`/%s
    mkdir -p ./htex_Local
    ''' % ('%s/bin' % pyenv_relpath,
           '%s/lib/python3.6/site-packages' % pyenv_relpath,
           x509_proxy)

    condor_cfg = '''
    transfer_output_files = htex_Local
    RequestMemory = %d
    RequestCpus = %d
    ''' % (mem_per_core * cores_per_job, cores_per_job)

    xfer_files = [pyenv_dir, '/tmp/%s' % (x509_proxy, )]

    condor_htex = Config(
        executors=[
            HighThroughputExecutor(
                label="coffea_parsl_condor_htex",
                address=address_by_hostname(),
                prefetch_capacity=0,
                cores_per_worker=1,
                max_workers=cores_per_job,
                worker_logdir_root='./',
                provider=CondorProvider(
                    channel=LocalChannel(),
                    init_blocks=24,
                    max_blocks=200,
                    nodes_per_block=1,
                    worker_init=wrk_init,
                    transfer_input_files=xfer_files,
                    scheduler_options=condor_cfg
                ),
            )
        ],
        strategy=None,
    )

    return condor_htex
