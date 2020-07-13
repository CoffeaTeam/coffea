import awkward1


def mixin_class(cls):
    """Decorator to register a mixin class

    Fills in any known behaviors based on class inheritance
    """
    name = cls.__name__
    awkward1.behavior[name] = type(name + "Record", (cls, awkward1.Record), {})
    awkward1.behavior["*", name] = type(name + "Array", (cls, awkward1.Array), {})
    for basecls in cls.__mro__:
        for method in basecls.__dict__.values():
            if hasattr(method, "_awkward_mixin"):
                ufunc, rhs, transpose = method._awkward_mixin
                if rhs is None:
                    awkward1.behavior.setdefault((ufunc, name), method)
                    continue
                for rhs_name in list(rhs) + [name]:
                    awkward1.behavior.setdefault((ufunc, name, rhs_name), method)
                    if transpose is not None:
                        awkward1.behavior.setdefault((ufunc, rhs_name, name), transpose)
                if basecls.__name__ in rhs:
                    rhs.add(name)
    return cls


def mixin_method(ufunc, rhs=None, transpose=True):
    """Decorator to register a mixin class method

    Using this decorator ensures that derived classes that are declared
    with the `mixin_class` decorator will also have the behaviors that this
    class has.

    ufunc : numpy.ufunc
        A universal function (or NEP18 callable) that is hooked in awkward1,
        i.e. it can be the first argument of a behavior
    rhs : Set[type] or None
        List of right-hand side argument types (leave None if unary function)
        The left-hand side is expected to always be ``self`` of the parent class
        If the function is not unary or binary, call for help :)
    transpose : bool
        Autmatically create a transpose signature (only makes sense for binary ufuncs)
    """

    def register(method):
        if not isinstance(rhs, (set, type(None))):
            raise ValueError("Expected a set of right-hand-side argument types")
        if transpose and rhs is not None:

            def transposed(left, right):
                return method(right, left)

            method._awkward_mixin = (ufunc, rhs, transposed)
        else:
            method._awkward_mixin = (ufunc, rhs, None)
        return method

    return register
