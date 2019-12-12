import six


def _mixin(methods, awkwardtype):
    '''Like awkward.Methods.mixin but also captures methods in dir() and propagate docstr'''
    newtype = type(methods.__name__ + 'Array', (methods, awkwardtype), {})
    newtype.__dir__ = lambda self: dir(methods) + awkwardtype.__dir__(self)
    if six.PY3:
        newtype.__doc__ = methods.__doc__
    return newtype
