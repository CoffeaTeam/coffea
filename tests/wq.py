import sys
import os
import os.path as osp


from coffea import hist, processor

try:
    import work_queue as wq
    work_queue_port = 9123
except ImportError:
    print("work_queue is not installed. Omiting test.")
    sys.exit(0)

def template_analysis(environment_file, filelist, executor, flatten, compression):
    from coffea.processor import run_uproot_job
    treename = 'Events'
    from coffea.processor.test_items import NanoTestProcessor

    exe_args = {
        'flatten': flatten,
        'compression': compression,
        'environment-file': environment_file,
        'resources-mode': 'fixed',
        'cores': 2,
        'memory': 500,  # MB
        'disk': 1000,   # MB
        'master-name': 'coffea-test',
        'port': work_queue_port,
        'print-stdout' : True
    }

    hists = run_uproot_job(filelist, treename, NanoTestProcessor(), executor, executor_args = exe_args)

    print(hists)
    assert(hists['cutflow']['ZJets_pt'] == 18)
    assert(hists['cutflow']['ZJets_mass'] == 6)
    assert(hists['cutflow']['Data_pt'] == 84)
    assert(hists['cutflow']['Data_mass'] == 66)

def work_queue_example(environment_file):
    from coffea.processor import work_queue_executor

    # Work Queue does not allow absolute paths
    filelist = {
        'ZJets': ['./samples/nano_dy.root'],
        'Data': ['./samples/nano_dimuon.root']
    }

    workers = wq.Factory(batch_type='local', master_host_port='localhost:{}'.format(work_queue_port))
    workers.max_workers = 1
    workers.min_workers = 1
    workers.cores  = 4
    workers.memory = 1000  # MB
    workers.disk   = 4000  # MB

    with workers:
        #template_analysis(environment_file, filelist, work_queue_executor, flatten=False, compression=0)
        #template_analysis(environment_file, filelist, work_queue_executor, flatten=True, compression=0)
        #template_analysis(environment_file, filelist, work_queue_executor, flatten=False, compression=2)
        template_analysis(environment_file, filelist, work_queue_executor, flatten=True, compression=2)

def create_conda_environment(env_file, py_version):
    """ Generate a conda environment file 'env_file' to send along the tasks. """

    if os.path.exists(env_file):
        print("conda environment file '{}' already exists. Not generating again.".format(env_file))
        return

    print("creating conda environment file '{}'...".format(env_file))
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w') as conda_recipe, tempfile.TemporaryDirectory() as tmp_env:
        conda_recipe.write("""
#! /bin/bash
set -e
conda create -y --prefix {tmp_env} python={py_version} conda six dill
source {tmp_env}/bin/activate
conda install -y -c conda-forge xrootd conda-pack
pip install ..
python -c 'import conda_pack; conda_pack.pack(prefix="{tmp_env}", output="{env_file}")'
""".format(env_file=env_file, tmp_env=tmp_env, py_version=py_version))

        conda_recipe.flush()

        subprocess.check_call(['/bin/bash', conda_recipe.name])
        print("done creating conda environment '{}'".format(env_file))


if __name__ == '__main__':
    py_version = "{}.{}".format(sys.version_info[0], sys.version_info[1])  # 3.6 or 3.7, or etc.

    environment_file = 'conda-coffea-wq-env-py{}.tar.gz'.format(py_version)

    create_conda_environment(environment_file, py_version)

    work_queue_example(environment_file)

