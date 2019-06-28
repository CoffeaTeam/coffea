Quickstart
==========

To try coffea now (without installing any code) experiment with our `hosted tutorial notebooks <https://mybinder.org/v2/gh/CoffeaTeam/coffea/master?filepath=binder/>`_.


Installation
------------

coffea is available on PyPI, Python 3.6+ is preferred since it supports all features of coffea, but a subset of coffea features will run in Python 2.7+. All functional features in each python version are routinely tested. You can see the python version you have installed by typing the following at the command prompt:

>>> python3 --version

coffea core functionality (i.e. everything but parsl and spark scale-out) has been tested on Windows, Linux and MacOS.
The parsl and spark scale-out options are tested to work with Linux and MacOS.

.. note:: coffea starts from v0.5.0 in the PyPI repository since before v0.5.0 it was hosted as `fnal-column-analysis-tools <https://pypi.org/project/fnal-column-analysis-tools/>`_. If you are still using fnal-column-analysis-tools, please move to `coffea <https://pypi.org/project/coffea/>`_!

Installation using Pip
^^^^^^^^^^^^^^^^^^^^^^

While ``pip`` and ``pip3`` can be used to install coffea we suggest the following approach
for reliable installation when many Python environments are avaialble.

1. Install coffea::

     $ python3 -m pip install coffea

To update a previously installed coffea to a newer version, use: ``python3 -m pip install -U coffea``

2. Install Jupyter for Tutorial notebooks::

     $ python3 -m pip install jupyter


.. note:: For more detailed info on setting up Jupyter with Python3.6 go `here <https://jupyter.readthedocs.io/en/latest/install.html>`_


Installation of Optional Packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

coffea supports several optional components that require additional module installations.
For example support for Apache `Spark <https://spark.apache.org/>`_, or parsl scale-out engines require additional packages that
can be installed easily via ``pip`` using a pip extras option.

Here's a list of the components and their extras option:

* Apache `Spark <https://spark.apache.org/>`_ job launcher : `run_spark_job`
* `parsl <https://github.com/Parsl/parsl/>`_ job launcher: `run_parsl_job`

Optional extras can be installed using the following syntax::

     $ python3 -m pip install coffea[<optional_package1>, <optional_package2>]

For Developers
--------------

1. Download coffea::

    $ git clone https://github.com/CoffeaTeam/coffea

2. Install::

    $ cd coffea
    $ pip install .
    ( To install specific extra options from the source :)
    $ pip install .[<optional_package1>...]

3. Start writing and doing physics analysis with coffea!

4. Test before committing some fancy new feature you made to coffea::

    $ python setup.py flake8
    $ python setup.py pytest

Requirements
------------

coffea requires the following:

* Python 2.7, 3.6+

For testing:

* pytest
* coverage

For building documentation:

* nbsphinx
* sphinx
* sphinx_rtd_theme