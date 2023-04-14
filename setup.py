#!/usr/bin/env python

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

from setuptools import find_packages, setup


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


INSTALL_REQUIRES = [
    "awkward>=2.1.3",
    "uproot>=5.0.7",
    "dask[array]>=2022.12.1,<2023.4.0",
    "dask-awkward>=2023.4.1",
    "dask-histogram>=2023.4.1",
    "correctionlib>=2.0.0",
    "pyarrow>=6.0.0",
    "fsspec",
    "matplotlib>=3",
    "numba>=0.56.0",
    "numpy>=1.22.0,<1.24",  # < 1.24 for numba 0.56 series
    "scipy>=1.1.0",
    "tqdm>=4.27.0",
    "lz4",
    "cloudpickle>=1.2.3",
    "toml>=0.10.2",
    "mplhep>=0.1.18",
    "packaging",
    "pandas",
    "hist>=2",
    "cachetools",
]
EXTRAS_REQUIRE = {}
EXTRAS_REQUIRE["spark"] = ["ipywidgets", "pyspark>=3.3.0", "jinja2"]
EXTRAS_REQUIRE["parsl"] = ["parsl>=2022.12.1"]
EXTRAS_REQUIRE["dask"] = [
    "dask[dataframe]>=2022.12.1",
    "distributed>=2022.12.1",
    "bokeh>=1.3.4",
    "blosc",
]
EXTRAS_REQUIRE["servicex"] = [
    "aiostream",
    "tenacity",
    "servicex>=2.5.3",
    "func-adl_servicex",
]
EXTRAS_REQUIRE["dev"] = [
    "pre-commit",
    "flake8",
    "black",
    "pytest",
    "pytest-cov",
    "pytest-mpl",
    "pytest-asyncio",
    "pytest-mock",
    "sphinx",
    "nbsphinx",
    "sphinx-rtd-theme",
    "sphinx-automodapi",
    "sphinx-copybutton>=0.3.2",
    "pyinstrument",
    "ipython",
]

setup(
    version=get_version(),
    packages=find_packages(exclude=["tests"]),
    scripts=[],
    include_package_data=True,
    long_description=get_description(),
    test_suite="tests",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
