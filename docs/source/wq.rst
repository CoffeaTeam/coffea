.. _intro-coffea-wq:

Work Queue Executor
===================

`Work Queue <https://cctools.readthedocs.io/en/latest/work_queue/>`_ is a
distributed computing framework used to build large scale master-worker
applications, developed by the Cooperative Computing Lab
(CCL) at the University of Notre Dame. This executor functions as the
master program which divides up a Coffea data analysis workload into
discrete tasks.  A large number of worker processes running on
cluster or cloud systems will execute the tasks.

To set up Coffea and Work Queue together, you will need to
create a Conda environment, install the software, and then
create a tarball containing the environment.  The tarball is
sent to each worker in order to provide the same environment
as the master machine.

.. code-block:: bash
  # Create a new environment
  conda create --name conda-coffea-wq-env
  conda activate conda-coffea-wq-env
  
  # Install Coffea and Work Queue into the environment
  conda install python=3.8.3 six dill
  conda install -c conda-forge ndcctools conda-pack xrootd
  pip install coffea
    
  # Pack the environment into a portable tarball.
  conda-pack --name conda-coffea-wq-env --output conda-coffea-wq-env.tar.gz

To run an analysis, you must set up a work queue executor
with appropriate arguments.  Here is a complete example that
builds upon the MyProcessor example from above.

.. literalinclude:: wq-example.py
   :language: Python

When executing this example,
you should see that Coffea begins to run, and a progress bar
shows the creation of tasks.  It is now waiting for worker
processes to connect and execute tasks.

For testing purposes, you can start a single worker on the same
machine, and direct it to connect to your master process.  With
a single worker, the process will be gradual as it completes
on task at a time.

>>> work_queue_worker -N coffea-wq-${USER}

To run at larger scale, you will need to run a large number
of workers on a cluster or other infrastructure.  For example,
to submit 32 workers to an HTCondor pool:

>>> condor_submit_workers -N coffea-wq-${USER} 32

For more information on starting and managing workers
on various batch systems and clusters, see the
`Work Queue <https://cctools.readthedocs.io/en/latest/work_queue/>` documentation
