fnal-column-analysis-tools
==========================

.. image:: https://travis-ci.com/CoffeaTeam/fnal-column-analysis-tools.svg?branch=master
    :target: https://travis-ci.com/CoffeaTeam/fnal-column-analysis-tools

.. image:: https://ci.appveyor.com/api/projects/status/93pk94vu1xlk83q5?svg=true
    :target: https://ci.appveyor.com/project/lgray/fnal-column-analysis-tools

.. image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/CoffeaTeam/fnal-column-analysis-tools/master?filepath=binder/

.. inclusion-marker-1-do-not-remove

Basic tools and wrappers for enabling not-too-alien syntax when running columnar Collider HEP analysis.

.. inclusion-marker-1-5-do-not-remove

This package is currently organized into three subpackages:

1) lookup_tools - This package manages importing corrections and scale factors, and provides a unified interface for evaluating those corrections on physics objects.
    - lookup_tools.extractor: handles importing the lookups from root files
    - lookup_tools.evaluator: handles organizing, providing an interface for, and evaluating the lookups
2) analysis_objects - This package contains definitions of physics objects casted in the language of JaggedArrays
    - JaggedCandidateArray - This object represents a list of candidates (things with four momenta and other attribute). Upon creation one can add extra columns of data that were not imported at construction, and all columns are accessible as though they were attributes of the class. This gives analysts a simple-to-read but rich, descriptive, and highly configurable object to represent muons, electrons, etc.
    - JaggedTLorentzVectorArray - This is a jagged representation of a TLorentzVectorArray. 
3) striped - This package defines transformations from the raw striped database into JaggedArrays and JaggedCandidateArrays
    - ColumnGroup - This object takes the name of a column that has attributes in striped and creates a dictionary of all given attributes.
    - PhysicalColumnGroup - Just like ColumnGroup except it requires a "p4" attribute to be defined, and is specialized to aide in creating JaggedCandidateArrays
    - jaggedFromColumnGroup - This is a function that takes a column group and returns a JaggedArray if it is a normal column group, or a JaggedCandidateArray if given a PhysicalColumnGroup

.. inclusion-marker-2-do-not-remove

Installation
============

Install fnal-column-analysis-tools like any other Python package:

.. code-block:: bash

    pip install fnal-column-analysis-tools

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

.. inclusion-marker-3-do-not-remove

Tutorial
========

This library is installed by people doing collider HEP analysis in the FNAL CMS group (so far).

Reference documentation
=======================

(...)
