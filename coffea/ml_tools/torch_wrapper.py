import abc
import warnings
from typing import Tuple, Union

import awkward
import dask_awkward
import numpy

try:
    import torch
except ImportError as err:
    warnings.warn(
        "Users should make sure the torch package is installed before proceeding!",
        ImportError,
    )
    raise err

from .helper import convert_args_pair


class torch_wrapper(abc.ABC):
    """
    Wrapper for running pytorch with awkward/dask-awkward inputs.
    """

    def __init__(
        self,
        torch_model: torch.nn.Module,
        torch_state: str,
        torch_device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Since pytorch models are directly pickle-able, we do not need to invoke
        lazy objects for this class.

        Parameters
        ----------

        - torch_model: The torch model object that will be used for inference

        - torch_state: Model state. Since this needs to be loaded after
          determining the torch computation device, this needs to be passed
          along with the model of interest.

        - torch_device: string representing the computation device to run
          inference on. ("cpu" or "gpu")
        """
        # Setting up the default torch inference computation device
        self.device = torch.device(torch_device)

        # Copy the pytorch model, loading in the state, and set to evaluation mode
        self.torch_model = torch_model
        self.torch_model.load_state_dict(
            torch.load(torch_state, map_location=torch.device(self.device))
        )
        self.torch_model.eval()

    def _validate_numpy_format(self, *args: numpy.array, **kwargs: numpy.array) -> None:
        """
        Checking where the input arguments are compatible with the loaded model.
        """
        # TODO: How to extract this information from just model?

    def _numpy_call(self, *args: numpy.array, **kwargs: numpy.array) -> numpy.array:
        """
        Evaluating the numpy inputs via the model. Returning the results also as
        as numpy array.
        """
        args = [torch.from_numpy(arr).to(self.device) for arr in args]
        kwargs = {k: torch.from_numpy(arr).to(self.device) for k, arr in kwargs.items()}
        with torch.no_grad():
            return self.torch_model(*args, **kwargs).detach().numpy()

    @abc.abstractclassmethod
    def awkward_to_numpy(self, *args: awkward.Array, **kwargs: awkward.Array) -> Tuple:
        """
        Abstract method to convert arbitrary awkward array inputs into the
        desired numpy format. The return should be a (*args, **kwargs)
        compatible pair that can be passed to the `_numpy_call` method.
        """
        raise NotImplementedError(
            "This needs to be overloaded by users! "
            "Consult the following documentation to find the awkward operations needed."
            "https://awkward-array.org/doc/main/user-guide/how-to-restructure-pad.html"
        )

    def __call_awkward__(
        self, *args: awkward.Array, **kwargs: awkward.Array
    ) -> awkward.Array:
        """
        Wrapper for a awkward array evaluation. Notice that the (*args,
        **kwargs) inputs for this function should match was is needed for the
        `awkward_to_numpy` method.
        """
        np_args, np_kwargs = self.awkward_to_numpy(*args, **kwargs)
        return awkward.from_regular(self._numpy_call(*np_args, **np_kwargs))

    def dask_touch(self, *args: awkward.Array, **kwargs: awkward.Array) -> None:
        """
        Given the input arrays in the format that will be used for
        awkward_to_numpy format conversion, touching branches of interest. For
        collection-like arrays (events/jets... etc)This would typically be in be
        something like `arr.<var>.layout._touch_data(recursive=True)` or
        `arr.<nested_collection>.<var>.layout._touch_data(recursive=True)`.

        For base implementation, we provide a method that is guaranteed to work
        but sacrifices performance: recursively touching all branching in the
        given array.
        """
        warnings.warn(
            "Default implementation is not lazy! "
            "Users should overload! with branches of interest",
            DeprecationWarning,
        )
        for arr in [*args, *kwargs.values()]:
            arr.layout._touch_data(recursive=True)

    def __call_dask__(
        self, *args: dask_awkward.Array, **kwargs: dask_awkward.Array
    ) -> dask_awkward.Array:
        """
        Wrapper for dask_awkward array evaluation. Notice that the (*args,
        **kwargs) inputs for this function should match was is needed for the
        `awkward_to_numpy` method.
        """

        class _callable_wrap(convert_args_pair):
            def __init__(self, inputs: Tuple, twrap: torch_wrapper):
                super().__init__(inputs)
                self.twrap = twrap

            def __call__(self, *args):
                # Call function can only recieve args
                if awkward.backend(args[0]) == "typetracer":
                    # For meta-data extraction
                    eval_args, eval_kwargs = self.args_to_pair(*args)
                    self.twrap.dask_touch(*eval_args, **eval_kwargs)

                    eval_args, eval_kwargs = self.args_to_pair(
                        *tuple(
                            v.layout.form.length_one_array(behavior=v.behavior)
                            for v in args
                        )
                    )
                    out = self.twrap.__call_awkward__(*eval_args, **eval_kwargs)

                    return awkward.Array(
                        out.layout.to_typetracer(forget_length=True),
                        behavior=out.behavior,
                    )
                else:
                    eval_args, eval_kwargs = self.args_to_pair(*args)
                    return self.twrap.__call_awkward__(*eval_args, **eval_kwargs)

        wrap = _callable_wrap((args, kwargs), self)

        return dask_awkward.lib.core.map_partitions(
            wrap,
            *wrap.pair_to_args(*args, **kwargs),
            label="torch_wrapper_dak",
            opt_touch_all=True,
        )

    def __call__(
        self,
        *args: Union[awkward.Array, dask_awkward.Array],
        **kwargs: Union[awkward.Array, dask_awkward.Array],
    ) -> Union[awkward.Array, dask_awkward.Array]:
        """
        Highest level abstraction to be directly called by the user. Notice that
        the input (*args, **kwargs) should match what is expected for the
        `awkward_to_numpy` method overloaded by the user.
        """
        all_arrs = [*args, *kwargs.values()]
        if all(isinstance(arr, awkward.Array) for arr in all_arrs):
            return self.__call_awkward__(*args, **kwargs)
        elif all(isinstance(arr, dask_awkward.Array) for arr in all_arrs):
            return self.__call_dask__(*args, **kwargs)
        else:
            raise ValueError(
                "All input arrays should be of the same type "
                "(either awkward.Array or dask_awkward.Array)"
            )
