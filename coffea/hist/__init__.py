"""Histogramming tools

`coffea.hist` is a histogram filling, transformation, and plotting sub-package, utilizing
numpy arrays for storage and matplotlib plotting routines for visualization.

Features found in this package are similar to those found in
packages such as `histbook <https://github.com/scikit-hep/histbook>`__ (deprecated),
`boost-histogram <https://github.com/scikit-hep/boost-histogram>`__ (in development),
`physt <https://github.com/scikit-hep/boost-histogram>`__, and built-in numpy
`histogram <https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html>`__ utilities.

"""
from .hist_tools import (
    Hist,
    Bin,
    Interval,
    Cat,
    StringBin,
)
from .plot import (
    poisson_interval,
    clopper_pearson_interval,
    normal_interval,
    plot1d,
    plotratio,
    plot2d,
    plotgrid,
)
from .export import export1d

__all__ = [
    'Hist',
    'Bin',
    'Interval',
    'Cat',
    'StringBin',
    'poisson_interval',
    'clopper_pearson_interval',
    'normal_interval',
    'plot1d',
    'plotratio',
    'plot2d',
    'plotgrid',
    'export1d'
]
