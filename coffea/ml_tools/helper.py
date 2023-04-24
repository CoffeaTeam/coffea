import abc
import warnings
from typing import List, Tuple


class convert_args_pair:
    """
    Helper class that helps with the conversion of the (*args, **kwargs) input
    pairs to *args and vice versa. Useful for interacting with map-partitions
    call wrapper, as those can only accept *args-like inputs.
    """

    def __init__(self, inputs: Tuple):
        """Storing the conversion dimension"""
        assert len(inputs) == 2
        self.args_len = len(inputs[0])
        self.kwargs_keys = list(inputs[1].keys())

    def args_to_pair(self, *args) -> Tuple:
        """Converting *args to a (*args,**kwargs) pair"""
        ret_args = tuple(x for x in args[0 : self.args_len])
        ret_kwargs = {k: v for k, v in zip(self.kwargs_keys, args[self.args_len :])}
        return ret_args, ret_kwargs

    def pair_to_args(self, *args, **kwargs) -> Tuple:
        """Converting (*args,**kwargs) pair to *args-like"""
        return Tuple(*args, *kwargs.values())


class lazy_container:
    """
    Generalizing the lazy object container syntax.

    For a given lazy object of name "obj", on container "x" initialization, it
    will create a dummy instance of the "x._obj = None". On the first time that
    "x.obj" is called, the contents of "x._obj" will be replaced by the return
    value of "x._create_obj()" method (this method cannot expect have additional
    input).

    Parameters
    ----------
        lazy_list : List[str]
            A list of string for the names of the lazy objects.

    """

    def __init__(self, lazy_list: List[str]):
        self._lazy_list = lazy_list

        for name in lazy_list:
            assert (
                name.isidentifier()
            ), f"Requested variable {name} cannot be used as variable name"
            assert hasattr(
                self, "_create_" + name
            ), f"Method _create_{name} needs to be implemented in class"
            setattr(self, "_" + name, None)

    def __getattr__(self, name):
        if name in self._lazy_list:
            if getattr(self, "_" + name) is None:
                setattr(self, "_" + name, getattr(self, "_create_" + name)())
            return getattr(self, "_" + name)
        else:
            return super().__getattr__(name)

    def __getstate__(self):
        """
        Explicitly setting all the lazy objects to be hidden from getstate
        requests. This ensures that wrapper objects can be pickled regardless of
        whether evaluations of the various objects has been carried out or not.
        """
        state = self.__dict__.copy()
        for name in self._lazy_list:
            state["_" + name] = None
        return state
