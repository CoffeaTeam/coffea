import warnings

import numpy

_tf_import_error = None
try:
    import tensorflow
except (ImportError, ModuleNotFoundError) as err:
    _tf_import_error = err

from .helper import nonserializable_attribute, numpy_call_wrapper


class tf_wrapper(nonserializable_attribute, numpy_call_wrapper):
    """
    Wrapper for running tensorflow inference with awkward/dask-awkward inputs.
    """

    def __init__(self, tf_model: str):
        """
        As models are not guaranteed to be directly serializable, the use will
        need to pass the model as files saved using the `tf.keras.save` method
        [1]. If the user is attempting to run on the clusters, the model file
        will need to be passed to the worker nodes in a way which preserves the
        file path.

        [1]
        https://www.tensorflow.org/guide/keras/serialization_and_saving#saving

        Parameters ----------

        - tf_model: Path to the tensorflow model file to load
        """
        if _tf_import_error is not None:
            warnings.warn(
                "Users should make sure the torch package is installed before proceeding!\n"
                "> pip install tensorflow\n"
                "or\n"
                "> conda install tensorflow",
                UserWarning,
            )
            raise _tf_import_error

        nonserializable_attribute.__init__(self, ["model", "device"])
        self.tf_model = tf_model

    def _create_device(self):
        """
        TODO: is this needed?
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_model(self):
        """
        Loading in the model from the model file. Tensorflow automatically
        determines if GPU are available or not and load the resources
        accordingly.
        """
        return tensorflow.keras.models.load_model(self.tf_model)

    def validate_numpy_input(self, *args: numpy.array, **kwargs: numpy.array) -> None:
        # Pytorch's model.parameters is not a reliable way to extract input
        # information for arbitrary models, so we will leave this to the user.
        super().validate_numpy_input(*args, **kwargs)

    def numpy_call(self, *args: numpy.array, **kwargs: numpy.array) -> numpy.array:
        """
        Evaluating the numpy inputs via the model. Here we are assuming all
        inputs can be trivially passed to the underlying model instance after a trivial
        `tensorflow.convert_to_tensor method`.
        """
        args = [
            (
                tensorflow.convert_to_tensor(arr)
                if arr.flags["WRITEABLE"]
                else tensorflow.convert_to_tensor(numpy.copy(arr))
            )
            for arr in args
        ]
        kwargs = {
            key: (
                tensorflow.convert_to_tensor(arr)
                if arr.flags["WRITABLE"]
                else tensorflow.convert_to_tensor(numpy.copy(arr))
            )
            for key, arr in kwargs.items()
        }
        return self.model(*args, **kwargs).numpy()
