# For python niceties
import abc
import warnings
from typing import Dict, List, Optional, Union

import awkward
import dask_awkward

# For data manipulation
import numpy

# For triton specific handling
try:
    import tritonclient.grpc
    import tritonclient.http
    import tritonclient.utils
except ImportError as err:
    warnings.warn(
        "Users should make sure the tritonclient package is installed before proceeding!",
        ImportError,
    )
    raise err

from .helper import convert_args_pair, lazy_container


class triton_wrapper(abc.ABC, lazy_container):
    """
    Wrapper for running triton inference.

    The target of this class is such that all triton specific operations are
    wrapped and abstracted-away from the users. The users should then only needs
    to handle awkward-level operations to mangle the arrays into the expected
    input format required by the the model of interest.

    The users should only need to interact with the following methods:

    - The constructor: provide a URL used to indicate the triton communication
      protocol to the triton server as well as the model of interest.
    - Overloading the `awkward_to_numpy` method (and alternatively the
      `dask_touch` method): manipulate some awkward array inputs into a numpy
      format suitable for the model of interest. When debugging this method, one
      can pass the output of this method to the `_validate_numpy_format`, where
      it compares the input format with the metadata of the model currently
      hosted on the triton server to check for format consistency.
    - `__call__`: the primary inference method, where the user passes over the
      arrays of interest, and the handling of numpy/awkward/dask_awkward types
      is handled automatically.
    """

    def __init__(
        self, model_url: str, client_args: Optional[Dict] = None, batch_size=-1
    ):
        """
        Parameters
        ----------

        - model_url: A string in the format of:
          triton+<protocol>://<address>/<model>/<version>

        - client_args: optional keyword arguments to pass to the underlying
          `InferenceServerClient` objects.

        - batch_size: How the input arrays should be split up for analysis
          processing. Leave negative to have this automatically resolved.
        """
        super(lazy_container, self).__init__(
            ["client", "model_metadata", "model_inputs", "model_outputs"]
        )

        fullprotocol, location = model_url.split("://")
        _, self.protocol = fullprotocol.split("+")
        self.address, self.model, self.version = location.split("/")

        # Additional pseudo-lazy objects that requires additional parsing after
        # lazy objects have been initialized or additional parsing.
        self._batch_size = batch_size
        self._client_args = client_args

    """
    Lazy object creation
    """

    @property
    def pmod(self):
        """Getting the protocol module based on the url protocol string."""
        if self.protocol == "grpc":
            return tritonclient.grpc
        elif self.protocol == "http":
            return tritonclient.http
        else:
            raise ValueError(
                f"{self.protocol} does not encode a valid protocol (grpc or http)"
            )

    def _create_client(self):
        return self.pmod.InferenceServerClient(url=self.address, **self.client_args)

    def client_args(self) -> Dict:
        """
        Function for adding default arguments to the client constructor kwargs.
        """
        if self.protocol == "grpc":
            kwargs = dict(verbose=False, ssl=True)
        elif self.protocol == "http":
            kwargs = dict(verbose=False, concurrency=12)
        if self._client_args is not None:
            kwargs.update(self.client_args)
        return kwargs

    def _create_metadata(self) -> Dict:
        return self.client.get_model_metadata(self.model, self.version, as_json=True)

    def _create_model_inputs(self) -> Dict[str, Dict]:
        """
        Extracting the model input data formats from the model_metatdata. Here
        we slightly change the input formats the objects in a format that is
        easier to manipulate and compare with numpy arrays.
        """
        return {
            x["name"]: {
                "shape": tuple(int(i) for i in x["shape"]),
                "datatype": x["datatype"],
            }
            for x in self.model_metadata["inputs"]
        }

    def _create_model_outputs(self) -> List[int]:
        """Getting a list of names of possible outputs"""
        return [x["name"] for x in self.model_metadata["outputs"]]

    @property
    def batch_size(self) -> int:
        """
        Getting the batch size to be used for array splitting. If it is
        explicitly set by the users, use that; otherwise, extract from the model
        configuration hosted on the server.
        """
        if self._batch_size < 0:
            model_config = self.client.get_model_config(
                self.model, self.version, as_json=True
            )["config"]
            if "dynamic_batching" in model_config:
                self._batch_size = model_config["dynamic_batching"][
                    "preferred_batch_size"
                ][0]
            else:
                self._batch_size = model_config["max_batch_size"]

        return self._batch_size

    """
    Numpy/awkward/dask_awkward inference
    """

    def _validate_numpy_format(
        self, output_list: List[str], input_dict: Dict[str, numpy.array]
    ) -> None:
        """
        Helper function for validating the numpy format to be passed to the
        inference request. This function will raise exceptions if any format is
        found to be wrong, with the error messages attempts to be as verbose and
        as descriptive as possible.
        """
        # Input value checking
        for iname, iarr in input_dict.items():
            # Checking the name
            if iname not in self.model_inputs.keys():
                raise ValueError(
                    f"Input '{iname}' not defined in model! "
                    f"Inputs defined by model: {[x for x in self.model_inputs.keys()]}"
                )
            # Checking the shape
            ishape = numpy.array(iarr.shape)
            mshape = numpy.array(self.model_inputs[iname]["shape"])
            if len(ishape) != len(mshape):
                raise ValueError(
                    f"Input {iname} got wrong dimension: {len(ishape)} "
                    f"(Expected {len(mshape)})"
                )
            if not all(numpy.where(mshape > 0, ishape == mshape, True)):
                raise ValueError(
                    f"Input {iname} got array of shape {ishape} "
                    f"(Expected: {mshape}, -1 means arbitrary)"
                )
            # Checking data type. Notice that this will only raise a warning! Data
            # type defined by triton can be found here:
            # https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#datatypes
            itype = iarr.dtype
            mtype = tritonclient.utils.triton_to_np_dtype(
                self.model_inputs[iname]["datatype"]
            )
            if itype != mtype:
                warnings.warn(
                    f"Input {iname} got array of type {itype} (Expected {mtype.__name__})."
                    " Automatic conversion will be performed using numpy.array.astype.",
                    UserWarning,
                )

        # Checking for missing inputs
        for mname in self.model_inputs.keys():
            if mname not in input_dict.keys():
                raise ValueError(f"Input {mname} not given in input dictionary!")

        # Checking output
        for oname in output_list:
            if oname not in self.model_outputs:
                raise ValueError(
                    f"Requested output {oname} not defined by model (Defined: {[x for x in self.model_outputs]})"
                )

    def _numpy_call(
        self, output_list: List[str], input_dict: Dict[str, numpy.array], validate=False
    ) -> Dict[str, numpy.array]:
        """
        The thinnest inferences request wrapping. Notice that this method should
        never be directly called other than for debugging purposes.

        Inputs to the inferences request should be in the format of a dictionary
        with the input-names as the dictionary key and the appropriate numpy
        array as the dictionary value. This input dictionary is automatically
        translated into a list of `tritonclient.InferInput` objects.

        Requested output should be a list of string, corresponding to the name
        of the outputs of interest. This strings will be automatically
        translated into the required `tritonclient.InferRequestedOutput`
        objects.

        The validate option will take the input/output requests, and compare it
        with the expected input/output as seen by the model_metadata hosted at
        the server, since this operation is slow it will be turned off by
        default, but it will be useful for debugging when developing the
        user-level `awkward_to_numpy` method.

        The return will be the dictionary of numpy arrays that have the
        output_list arguments as keys.
        """
        if validate:
            self._validate_numpy_format(output_list, input_dict)

        # Setting up the inference input containers
        def _get_infer_shape(name):
            ishape = numpy.array(input_dict[name].shape)
            mshape = numpy.array(self.model_inputs[name]["shape"])
            mshape = numpy.where(mshape < 0, ishape, mshape)
            mshape[0] = self.batch_size
            return mshape

        infer_inputs = [
            self.pmod.InferInput(name, _get_infer_shape(name), prop["datatype"])
            for name, prop in self.model_inputs.items()
        ]

        # Setting up the inference output containers
        infer_outputs = [
            self.pmod.InferRequestedOutput(output) for output in output_list
        ]

        # Setting up container for storing output.
        output = None

        # Padding the outermost dimension to a multiple of of the batch size
        orig_len = list(input_dict.values())[0].shape[0]  # saving original length
        for start_idx in range(0, orig_len, self.batch_size):
            stop_idx = min([start_idx + self.batch_size, orig_len])

            for idx, name in enumerate(self.model_inputs.keys()):
                mtype = tritonclient.utils.triton_to_np_dtype(
                    self.model_inputs[name]["datatype"]
                )
                shape = list(input_dict[name].shape)
                shape[0] = self.batch_size  # Always pad to fixed length
                infer_inputs[idx].set_data_from_numpy(
                    numpy.resize(
                        input_dict[name][start_idx:stop_idx],  # We need a copy here
                        tuple(shape),
                    ).astype(mtype)
                )

            # Making request to server
            request = self.client.infer(
                self.model,
                model_version=self.version,
                inputs=infer_inputs,
                outputs=infer_outputs,
            )
            if output is None:
                output = {
                    o: request.as_numpy(o)[start_idx:stop_idx] for o in output_list
                }
            else:
                for o in output_list:
                    output[o] = numpy.concatenate(
                        (output[o], request.as_numpy(o)), axis=0
                    )
        return {k: v[:orig_len] for k, v in output.items()}

    @abc.abstractclassmethod
    def awkward_to_numpy(
        self, *args: awkward.Array, **kwargs: awkward.Array
    ) -> Dict[str, numpy.array]:
        """
        Abstract method to convert arbitrary awkward array inputs into the
        desired numpy format. Given that there cannot be a fixed input format
        for inference models, here we provide common interface that users should
        overload to provide a simple way of abstracting the high-level
        functions. Since this method explicitly require the conversion to numpy
        arrays, the input is assumed to be eager/non-dask awkward arrays.
        """
        raise NotImplementedError(
            "This needs to be overloaded by users! "
            "Consult the following documentation to find the awkward operations needed."
            "https://awkward-array.org/doc/main/user-guide/how-to-restructure-pad.html"
        )

    def __call_awkward__(
        self, output_list: List[str], *args: awkward.Array, **kwargs: awkward.Array
    ) -> awkward.Array:
        """
        Translation function for running inference on awkward arrays. The method
        for translating the awkward array inputs into numpy arrays will be
        automatically invoked.
        """
        return {
            k: awkward.from_numpy(v)
            for k, v in self._numpy_call(
                output_list, self.awkward_to_numpy(*args, **kwargs)
            ).items()
        }

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
        self,
        output_list: List[str],
        *args: dask_awkward.Array,
        **kwargs: dask_awkward.Array,
    ) -> dask_awkward.Array:
        """
        Translation function for running inference on dask awkward arrays.
        Wrapping the `__call_awkward__` method into a callable object that is
        suitable to be dask_awkward.map_partition method.
        """

        class _callable_wrap(convert_args_pair):
            def __init__(self, inputs, twrap, output_list):
                """
                Here we need to also store the args_len and keys argument, as
                the map_partition method currently only works with *args like
                arguments. These containers are needed to properly translate the
                passed *args to a (*args, **kwargs) pair used by
                __call_awkward__.
                """
                super().__init__(inputs)
                self.twrap = twrap
                self.output_list = output_list

            def __call__(self, *args):
                """
                Mainly translating the input *args to the (*args, **kwarg) pair
                defined for the `__call_awkward__` method. Additional
                calculation routine defined to for the 'typetracer' backend for
                metadata scouting.
                """
                if awkward.backend(args[0]) == "typetracer":
                    # None-recursive touching
                    # for arr in args:
                    #     arr.layout._touch_data(recursive=False)

                    # Running the touch overload function
                    eval_args, eval_kwargs = self.args_to_pair(*args)
                    self.twrap.dask_touch(*eval_args, **eval_kwargs)

                    # Getting the length-one array for evaluation
                    eval_args, eval_kwargs = self.args_to_pair(
                        *tuple(
                            v.layout.form.length_one_array(behavior=v.behavior)
                            for v in args
                        )
                    )

                    # awkward.zip so that the return is a single awkward
                    # array
                    out = awkward.zip(
                        self.twrap.__call_awkward__(
                            output_list, *eval_args, **eval_kwargs
                        )
                    )
                    return awkward.Array(
                        out.layout.to_typetracer(forget_length=True),
                        behavior=out.behavior,
                    )
                else:
                    eval_args, eval_kwargs = self.args_to_pair(*args)
                    return awkward.zip(
                        self.twrap.__call_awkward__(
                            output_list, *eval_args, **eval_kwargs
                        )
                    )

        wrap = _callable_wrap((args, kwargs), self, output_list)
        arr = dask_awkward.lib.core.map_partitions(
            wrap,
            *wrap.pair_to_args(*args, **kwargs),
            label="triton_wrapper_dak",
            opt_touch_all=True,
        )
        return {o: arr[o] for o in output_list}

    def __call__(
        self,
        output_list: List[str],
        *args: Union[awkward.Array, dask_awkward.Array],
        **kwargs: Union[awkward.Array, dask_awkward.Array],
    ) -> Union[awkward.Array, dask_awkward.Array]:
        """
        Highest level abstraction to be directly called by the user. The
        translation of the input arrays (either eager or dask), will be
        automatically detected. Notice that the input *args, **kwargs should
        match what is expected for the `awkward_to_numpy` method overloaded by
        the user.
        """
        all_arrs = [*args, *kwargs.values()]
        if all(isinstance(arr, awkward.Array) for arr in all_arrs):
            return self.__call_awkward__(output_list, *args, **kwargs)
        elif all(isinstance(arr, dask_awkward.Array) for arr in all_arrs):
            return self.__call_dask__(output_list, *args, **kwargs)
        else:
            raise ValueError(
                "All input arrays should be of the same type "
                "(either awkward.Array or dask_awkward.Array)"
            )
