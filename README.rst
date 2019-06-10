coffea
==========================

.. image:: https://travis-ci.com/CoffeaTeam/coffea.svg?branch=master
    :target: https://travis-ci.com/CoffeaTeam/coffea

.. image:: https://ci.appveyor.com/api/projects/status/co4wg4074jal3klq/branch/master?svg=true
    :target: https://ci.appveyor.com/project/lgray/coffea/branch/master

.. image:: https://codecov.io/gh/CoffeaTeam/coffea/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/CoffeaTeam/coffea

.. image:: https://badge.fury.io/py/coffea.svg
    :target: https://badge.fury.io/py/coffea

.. image:: https://img.shields.io/pypi/dm/coffea.svg
    :target: https://img.shields.io/pypi/dm/coffea

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/CoffeaTeam/coffea/master?filepath=binder/

.. inclusion-marker-1-do-not-remove

Basic tools and wrappers for enabling not-too-alien syntax when running columnar Collider HEP analysis.

.. inclusion-marker-1-5-do-not-remove

coffea is currently organized into several sub-modules with specific purposes.

 
1) analysis_objects - This package contains definitions of physics objects casted in the language of JaggedArrays

2) arrays - Another take on making analysis objects with directly decorated LorentzVector objects instead of wrapped LorentzVectors.

3) hist - A well-featured histogramming and plotting sub-package.

4) jetmet_tools - CMS-specific tools for correcting Jets and Missing Energy

5) lookup_tools - This package manages importing corrections and scale factors, and provides a unified interface for evaluating those corrections on physics objects.

6) lumi tools - A CMS-specific package for parsing luminosity database files to derive integrated luminosity and good run lists.

7) processor - An interface for defining and running analyses in a portable way across a variety of scale-out mechanisms.

8) striped - This package defines transformations from the raw striped database into JaggedArrays and JaggedCandidateArrays, but is somewhat deprecated.
    
For further information please see the complete package index in our `documentation <https://coffeateam.github.io/coffea/>`_.

.. inclusion-marker-2-do-not-remove

Installation
============

Install coffea like any other Python package:

.. code-block:: bash

    pip install coffea

or similar (use ``sudo``, ``--user``, ``virtualenv``, or pip-in-conda if you wish).

Strict dependencies:
====================

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`__ (2.7+, 3.6+)

The following are installed automatically when you install uproot with pip:

- `numpy <https://scipy.org/install.html>`__ (1.15+)
- `awkward-array <https://pypi.org/project/awkward>`__ to manipulate data from non-flat TTrees, such as jagged arrays (`part of Scikit-HEP <https://github.com/scikit-hep/awkward-array>`__)
- `uproot-methods <https://pypi.org/project/uproot-methods>`__ to allow expressions of things as lorentz vectors
- `numba <https://numba.pydata.org/>`__ just-in-time compilation of python functions
- ``scipy`` for statistical functions
- ``matplitlib`` as a plotting backend
- ``uproot`` for interacting with ROOT files
- ``tqdm``

.. inclusion-marker-3-do-not-remove

Tutorial
========

This library is installed by people doing collider HEP analysis in the FNAL CMS group (so far).

Reference documentation
=======================

Please read our documentation at: https://coffeateam.github.io/coffea/
