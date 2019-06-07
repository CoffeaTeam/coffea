try:
    import awkward.numba as awkward
except ImportError:
    import awkward

from awkward.util import numpy
import numba

akd = awkward
np = numpy
nb = numba

import lz4.frame
import cloudpickle


def load(filename):
    '''
    Load a coffea file from disk
    '''
    with lz4.frame.open(filename) as fin:
        output = cloudpickle.load(fin)
    return output


def save(output, filename):
    '''
    Save a coffea object or collection thereof to disk
    Suggested suffix: .coffea
    '''
    with lz4.frame.open(filename, 'wb') as fout:
        cloudpickle.dump(output, fout)
