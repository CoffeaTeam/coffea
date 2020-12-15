"""Utility functions

"""
import awkward
import hashlib
import numpy
import numba
import coffea

ak = awkward
np = numpy
nb = numba

import lz4.frame
import cloudpickle
import warnings


def load(filename):
    '''Load a coffea file from disk
    '''
    with lz4.frame.open(filename) as fin:
        output = cloudpickle.load(fin)
    return output


def save(output, filename):
    '''Save a coffea object or collection thereof to disk

    This function can accept any picklable object.  Suggested suffix: ``.coffea``
    '''
    with lz4.frame.open(filename, 'wb') as fout:
        thepickle = cloudpickle.dumps(output)
        fout.write(thepickle)


def _hex(string):
    try:
        return string.hex()
    except AttributeError:
        return "".join("{:02x}".format(ord(c)) for c in string)


def _ascii(maybebytes):
    try:
        return maybebytes.decode('ascii')
    except AttributeError:
        return maybebytes


def _hash(items):
    # python 3.3 salts hash(), we want it to persist across processes
    x = hashlib.md5(bytes(';'.join(str(x) for x in items), 'ascii'))
    return int(x.hexdigest()[:16], base=16)


def _ensure_flat(array, allow_missing=False):
    """Normalize an array to a flat numpy array or raise ValueError"""
    if not isinstance(array, (ak.Array, numpy.ndarray)):
        raise ValueError("Expected a numpy or awkward array, received: %r" % array)

    aktype = ak.type(array)
    if not isinstance(aktype, ak.types.ArrayType):
        raise ValueError("Expected an array type, received: %r" % aktype)
    isprimitive = isinstance(aktype.type, ak.types.PrimitiveType)
    isoptionprimitive = isinstance(aktype.type, ak.types.OptionType) and isinstance(aktype.type.type, ak.types.PrimitiveType)
    if allow_missing and not (isprimitive or isoptionprimitive):
        raise ValueError("Expected an array of type N * primitive or N * ?primitive, received: %r" % aktype)
    if not (allow_missing or isprimitive):
        raise ValueError("Expected an array of type N * primitive, received: %r" % aktype)
    if isinstance(array, ak.Array):
        array = ak.to_numpy(array, allow_missing=allow_missing)
    return array


# lifted from awkward - https://github.com/scikit-hep/awkward-1.0/blob/5fe31a916bf30df6c2ea10d4094f6f1aefcf3d0c/src/awkward/_util.py#L47-L61 # noqa
# drive our deprecations-as-errors as with awkward
def deprecate(exception, version, date=None):
    if coffea.deprecations_as_errors:
        raise exception
    else:
        if date is None:
            date = ""
        else:
            date = " (target date: " + date + ")"
        message = """In coffea version {0}{1}, this will be an error.
(Set coffea.deprecations_as_errors = True to get a stack trace now.)
{2}: {3}""".format(
            version, date, type(exception).__name__, str(exception)
        )
        warnings.warn(message, FutureWarning)
