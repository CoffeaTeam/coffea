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

supported_column_file_types = {b'root': 'root', b'PAR1': 'parquet'}


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


def get_column_file_type(filestr, timeout=10):
    bytes = None
    if filestr.startswith('root'):
        from uproot4.source.xrootd import XRootDResource
        resource = XRootDResource(filestr, timeout=timeout)
        bytes = resource.file.read(0, 4, timeout=timeout)[1]
        resource.file.close()
    elif filestr.startswith('http'):
        from uproot4.source.http import HTTPResource, make_connection, full_path
        import requests
        actual_url = None
        head = requests.head(filestr)
        while head.is_redirect:
            head = requests.head(head.next.url)
        actual_url = head.url
        resource = HTTPResource(actual_url, timeout=timeout)
        connection = make_connection(resource.parsed_url, timeout)
        connection.request(
            "GET",
            full_path(resource.parsed_url),
            headers={"Range": "bytes={0}-{1}".format(0, 3)},
        )
        bytes = resource.get(connection, 0, 4)
        connection.close()
    else:
        with open(filestr, 'rb') as file:
            bytes = file.read(4)

    type = None
    try:
        type = supported_column_file_types[bytes]
    except KeyError:
        raise Exception('Unsupported column source file type {0}.'
                        ' Not one of {1}'.format(bytes, list(supported_column_file_types.keys())))

    if filestr.startswith(('root', 'http')) and type == 'parquet':
        raise Exception('Reading parquet files does not yet support reads over xrootd to http(s)!')

    return type


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
