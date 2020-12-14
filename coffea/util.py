"""Utility functions

"""
import awkward
import awkward1
import hashlib
import numpy
import numba

akd = awkward
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
    if isinstance(array, awkward.AwkwardArray):
        array = awkward1.from_awkward0(array)
    elif not isinstance(array, (awkward1.Array, numpy.ndarray)):
        raise ValueError("Expected a numpy or awkward array, received: %r" % array)

    aktype = awkward1.type(array)
    if not isinstance(aktype, awkward1.types.ArrayType):
        raise ValueError("Expected an array type, received: %r" % aktype)
    isprimitive = isinstance(aktype.type, awkward1.types.PrimitiveType)
    isoptionprimitive = isinstance(aktype.type, awkward1.types.OptionType) and isinstance(aktype.type.type, awkward1.types.PrimitiveType)
    if allow_missing and not (isprimitive or isoptionprimitive):
        raise ValueError("Expected an array of type N * primitive or N * ?primitive, received: %r" % aktype)
    if not (allow_missing or isprimitive):
        raise ValueError("Expected an array of type N * primitive, received: %r" % aktype)
    if isinstance(array, awkward1.Array):
        array = awkward1.to_numpy(array, allow_missing=allow_missing)
    return array


# lifted from awkward1 - https://github.com/scikit-hep/awkward-1.0/blob/5fe31a916bf30df6c2ea10d4094f6f1aefcf3d0c/src/awkward/_util.py#L47-L61 # noqa
# we will drive the deprecations as errors using the awkward1 flag for it
# since this is largely an awkward1 related campaign
class Awkward0Warning(FutureWarning):
    pass


def deprecate(exception, version, date=None):
    if awkward1.deprecations_as_errors:
        raise exception
    else:
        if date is None:
            date = ""
        else:
            date = " (target date: " + date + ")"
        message = """In coffea version {0}{1}, this will be an error.
(Set awkward1.deprecations_as_errors = True to get a stack trace now.)
{2}: {3}""".format(
            version, date, type(exception).__name__, str(exception)
        )
        warnings.warn(message, Awkward0Warning)


# if we have found awkward0 being passed to coffea, complain
def deprecate_detected_awkward0(*args, **kwargs):
    has_ak0 = any([isinstance(arg, awkward.array.base.AwkwardArray) for arg in args])
    has_ak0 |= any([isinstance(arg, awkward.array.base.AwkwardArray) for arg in kwargs.values()])

    # special case, if no args or kwargs given just emit!
    if len(args) == len(kwargs) == 0:
        has_ak0 = True

    if has_ak0:
        e = TypeError('Use of awkward 0.x arrays in coffea is deprecated!\nIn coming releases'
                      ' coffea will only accept awkward > 1.0 arrays and awkward0-only classes'
                      ' will be removed.')
        deprecate(e, '0.7', date='January 2021')


def deprecate_awkward0_util(item):
    e = TypeError('{0} relies exclusively on awkward 0.x and will be removed in upcoming'
                  ' versions of coffea!'.format(item))
    deprecate(e, '0.7', date='January 2021')
