import six
import awkward


def _mixin(methods, awkwardtype):
    '''Like awkward.Methods.mixin but also captures methods in dir() and propagate docstr'''
    if not issubclass(awkwardtype, awkward.array.base.AwkwardArray):
        raise ValueError("Only mix in to awkward types or derived")
    if not issubclass(methods, awkward.array.objects.Methods):
        raise ValueError("Can only mixin methods deriving from awkward Methods ABC")
    newtype = type(methods.__name__ + 'Array', (methods, awkwardtype), {})
    newtype.__dir__ = lambda self: dir(methods) + awkwardtype.__dir__(self)
    if six.PY3:
        newtype.__doc__ = methods.__doc__
    return newtype
