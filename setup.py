#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, Fermilab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os.path
import six

from setuptools import find_packages
from setuptools import setup


def get_version():
    g = {}
    exec(open(os.path.join("coffea", "version.py")).read(), g)
    return g["__version__"]


def get_description():
    description = open("README.rst", "rb").read().decode("utf8", "ignore")
    start = description.index(".. inclusion-marker-1-5-do-not-remove")
    stop = description.index(".. inclusion-marker-3-do-not-remove")

    #    before = ""
    #    after = """
    # Reference documentation
    # =======================
    # """

    return description[start:stop].strip()  # before + + after


INSTALL_REQUIRES = ['awkward>=0.12.20',
                    'uproot-methods>=0.7.3',
                    'uproot>=3.11.0',
                    'awkward1>=0.2.32',
                    'uproot4>=0.0.17',
                    'matplotlib<3' if six.PY2 else 'matplotlib>=3',
                    'numba>=0.43.1',
                    'numpy>=1.16.0',
                    'scipy>=1.1.0',
                    'futures; python_version == "2.7"',
                    'tqdm>=4.27.0',
                    'lz4',
                    'six',
                    'cloudpickle',
                    'mplhep>=0.1.18',
                    'packaging',
                    'ipywidgets',
                    'pandas',
                    ]
EXTRAS_REQUIRE = {}
if six.PY3:
    templates = ['jinja2']
    EXTRAS_REQUIRE['spark'] = ['pyspark>=2.4.1,<3.0.0', 'pyarrow>=0.10.0,!=0.14.0'] + templates
    EXTRAS_REQUIRE['parsl'] = ['parsl>=0.7.2']
    EXTRAS_REQUIRE['dask'] = ['dask>=2.6.0', 'distributed>=2.6.0', 'bokeh>=1.3.4', 'blosc']
if six.PY2:
    EXTRAS_REQUIRE['striped'] = []

setup(name="coffea",
      version=get_version(),
      packages=find_packages(exclude=["tests"]),
      scripts=[],
      include_package_data=True,
      description="Tools for doing Collider HEP style analysis with columnar operations",
      long_description=get_description(),
      author="Lindsey Gray (Fermilab)",
      author_email="lagray@fnal.gov",
      maintainer="Lindsey Gray (Fermilab)",
      maintainer_email="lagray@fnal.gov",
      url="https://github.com/CoffeaTeam/coffea",
      download_url="https://github.com/CoffeaTeam/coffea/releases",
      license="BSD 3-clause",
      test_suite="tests",
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      setup_requires=["pytest-runner", "flake8"],
      tests_require=["pytest"],
      classifiers=[
          "Development Status :: 4 - Beta",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Operating System :: MacOS",
          "Operating System :: POSIX",
          "Operating System :: Unix",
          "Programming Language :: Python",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development",
          "Topic :: Utilities",
      ],
      platforms="Any",
      )
