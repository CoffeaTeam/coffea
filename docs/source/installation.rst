.. _installing-coffea:

Installing coffea
=================

Quick start
-----------
To try coffea now, without installing anything, you can experiment with our
`hosted tutorial notebooks <https://mybinder.org/v2/gh/CoffeaTeam/coffea/master?filepath=binder/>`_.

Platform support
----------------
Coffea is a python package distributed via `PyPI <https://pypi.org/project/coffea>`_. A python installation is required to use coffea.
Python version 3.6 or newer is preferred since it supports all features of coffea, but a subset of coffea features will run in python 2.7 or newer.

.. note:: Python 2 end-of-life is Jan. 1, 2020. All major scientific python packages will no longer provide support for python 2 by that date: https://python3statement.org/

All functional features in each supported python version are routinely tested.
You can see the python version you have installed by typing the following at the command prompt:

>>> python --version

or, in some cases, if both python 2 and 3 are available, you can find the python 3 version via:

>>> python3 --version

coffea core functionality is routinely tested on Windows, Linux and MacOS.
All :ref:`def-local-executors` are tested against all three platforms,
however the :ref:`def-distributed-executors` are not routinely tested on Windows.

Coffea starts from v0.5.0 in the PyPI repository since before v0.5.0 it was hosted as `fnal-column-analysis-tools <https://pypi.org/project/fnal-column-analysis-tools/>`_. If you are still using fnal-column-analysis-tools, please move to `coffea <https://pypi.org/project/coffea/>`_!

Install coffea
--------------
To install coffea, there are several mostly-equivalent options:

   - install coffea system-wide using ``pip install coffea``; 
   - if you do not have administrator permissions, install as local user with ``pip install --user coffea``;
   - if you prefer to not place coffea in your global environment, you can set up a `virtual environment <https://docs.python.org/3/library/venv.html>`_, and use the venv-provided pip;
   - if you use `Conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_,  simply activate the environment you wish to use and install via the conda-provided pip.
   - or, if you are using a machine that has cvmfs available, see `Install via cvmfs`_ below.

To update a previously installed coffea to a newer version, use: ``pip install --upgrade coffea``
Although not required, it is recommended to also `install Jupyter <https://jupyter.org/install>`_, as it provides a more interactive development environment.
The installation procedure is essentially identical as above: ``pip install jupyter``. (If you use conda, ``conda install jupyter`` is a better option.)

In rare cases, you may find that the ``pip`` executable in your path does not correspond to the same python installation as the ``python`` executable. This is a sign of a broken python environment. However, this can be bypassed by using the syntax ``python -m pip ...`` in place of ``pip ...``.

Install optional dependencies
-----------------------------
Coffea supports several optional components that require additional package installations.
In particular, all of the :ref:`def-distributed-executors` require additional packages.
The necessary dependencies can be installed easily via ``pip`` using the setuptools `extras <https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies>`_  facility:

   - Apache `Spark <https://spark.apache.org/>`_ distributed executor: ``pip install coffea[spark]``
   - `parsl <http://parsl-project.org/>`_ distributed executor: ``pip install coffea[parsl]``
   - `dask <https://distributed.dask.org/en/latest/>`_ distributed executor: ``pip install coffea[dask]``
   - `Work Queue <https://cctools.readthedocs.io/en/latest/work_queue/>`_ distributed executor: see :ref:`intro-coffea-wq` for installation instructions

Multiple extras can be installed together via, e.g. ``pip install coffea[dask,spark]``

Install via cvmfs
-----------------
Although the local installation can work anywhere, if the base environment does not already have most of the coffea dependencies, then the user-local package directory can become quite bloated.
An option to avoid this bloat is to use a base python environment provided via `CERN LCG <https://ep-dep-sft.web.cern.ch/document/lcg-releases>`_, which is available on any system that has the `cvmfs <https://cernvm.cern.ch/portal/filesystem>`_ directory ``/cvmfs/sft.cern.ch/`` mounted.
Simply source a LCG release (shown here: 96python3) and install:

.. code-block:: bash

  # check your platform: CC7 shown below, for SL6 it would be "x86_64-slc6-gcc8-opt"
  source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh  # or .csh, etc.
  pip install --user coffea

Creating a portable virtual environment
---------------------------------------
In some instances, it may be useful to have a self-contained environment that can be relocated.
One use case is for users of coffea that do not have access to a distributed compute cluster that is compatible with
one of the coffea distributed executors. Here, a fallback solution can be found by creating traditional batch jobs (e.g. condor)
which then use coffea local executors, possibly multi-threaded. In this case, often the user-local python package directory
is not available from batch workers, so a portable python enviroment needs to be created.
Annoyingly, python virtual environments are not portable by default due to several hardcoded paths in specific locations,
however there are not many locations and some sed hacks can save the day.
Here is an example of a bash script that installs coffea on top of the LCG 96python3 software stack inside a portable virtual environment,
with the caveat that cvmfs must be visible from batch workers:

.. code-block:: bash

  #!/usr/bin/env bash
  NAME=coffeaenv
  LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh

  source $LCG
  # following https://aarongorka.com/blog/portable-virtualenv/, an alternative is https://github.com/pantsbuild/pex
  python -m venv --copies $NAME
  source $NAME/bin/activate
  python -m pip install setuptools pip --upgrade
  python -m pip install coffea
  sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $NAME/bin/activate
  sed -i '1s/#!.*python$/#!\/usr\/bin\/env python/' $NAME/bin/*
  sed -i "2a source ${LCG}" $NAME/bin/activate
  tar -zcf ${NAME}.tar.gz ${NAME}

The resulting tarball size is about 5 MB.
An example batch job wrapper script is:

.. code-block:: bash

  #!/usr/bin/env bash
  tar -zxf coffeaenv.tar.gz
  source coffeaenv/bin/activate

  echo "Running command:" $@
  time $@ || exit $?
  
Unless you install jupyter into this environment (which may bloat the tarball--LCG96 jupyter is reasonably recent), it is not visible inside the LCG jupyter server. From a shell with the virtual environment activated, you can execute::

  python -m ipykernel install --user --name=coffeaenv

to make a new kernel available that uses this environment.

For Developers
--------------

1. Download source:

  .. code-block:: bash

    git clone https://github.com/CoffeaTeam/coffea

2. Install with development dependencies:

  .. code-block:: bash

    cd coffea
    pip install --editable .[dev]
    // or if you need to work on the executors, e.g. dask,
    pip install --editable .[dev,dask]

3. Develop a cool new feature or fix some bugs

4. Lint source, run tests, and build documentation:

  .. code-block:: bash

    flake8 coffea
    pytest tests
    pushd docs && make html && popd

5. Make a pull request!
