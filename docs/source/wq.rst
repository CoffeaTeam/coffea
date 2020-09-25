.. _intro-coffea-wq:

Work Queue
==========

`Work Queue <https://cctools.readthedocs.io/en/latest/work_queue/>`_ is a
production framework used to build large scale master-
worker applications, developed by the Cooperative Computing Laboratory
(CCL) at the University of Notre Dame. This executor functions as the
master program which submits chunks of data as tasks. A remote worker,
which can be run on cluster and cloud systems, will be able to execute
the task.

python_package_run is the necessary wrapper script. This script is
installed with work queue. The executor will try to find this wrapper in
PATH. Also, The location of this script can be specified with the 'wrapper'
argument.

To set up Work Queue, the following procedure can be used:

  .. code-block:: bash

    conda create --name conda-coffea-wq-env python=3.8 six dill
    conda activate conda-coffea-wq-env
    conda install -c conda-forge xrootd ndcctools
    pip install <path to coffea directory>
    conda activate base
    pip install conda-pack
    python -c 'import conda_pack; conda_pack.pack(name="conda-coffea-wq-env", output="conda-coffea-wq-env.tar.gz")'
    conda activate conda-coffea-wq-env
