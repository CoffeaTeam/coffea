try:
    import awkward.numba as awkward
except ImportError:
    import awkward

from awkward.util import numpy

import numba

akd = awkward
np = numpy
nb = numba
