fnal-column-analysis-tools
==========================

.. inclusion-marker-1-do-not-remove

Basic tools and wrappers for enabling not-to-alien syntax when running columnar Collider HEP analysis.

.. inclusion-marker-1-5-do-not-remove

(...)

.. inclusion-marker-2-do-not-remove

Installation
============

Install uproot-methods like any other Python package:

.. code-block:: bash

    pip install fnal-column-analysis-tools

or similar (use ``sudo``, ``--user``, ``virtualenv``, or pip-in-conda if you wish).

Strict dependencies:
====================

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`__ (2.7+, 3.4+)

The following are installed automatically when you install uproot with pip:

- `Numpy <https://scipy.org/install.html>`__ (1.13.1+)
- `awkward-array <https://pypi.org/project/awkward>`__ to manipulate data from non-flat TTrees, such as jagged arrays (`part of Scikit-HEP <https://github.com/scikit-hep/awkward-array>`__)
- `uproot-methods <https://pypi.org/project/uproot-methods>`__ to allow expressions of things as lorentz vectors

.. inclusion-marker-3-do-not-remove

Tutorial
========

This library is installed by people doing collider HEP analysis in the FNAL CMS group (so far).

Reference documentation
=======================

(...)