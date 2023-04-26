import warnings
import numpy

try:
    import torch
except ImportError as err:
    warnings.warn(
        "Users should make sure the torch package is installed before proceeding!\n"
        "> pip install torch\n"
        "or\n"
        "> conda install torch",
        UserWarning,
    )
    raise err

from .helper import numpy_call_wrapper


class torch_wrapper(numpy_call_wrapper):
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

    def validate_numpy_inputs(self, *args: numpy.array, **kwargs: numpy.array) -> None:
        # TODO: How to extract this information from just model?
        pass

    def numpy_call(self, *args: numpy.array, **kwargs: numpy.array) -> numpy.array:
        """
        Evaluating the numpy inputs via the model. Returning the results also as
        as numpy array.
        """
        args = [torch.from_numpy(arr).to(self.device) for arr in args]
        kwargs = {k: torch.from_numpy(arr).to(self.device) for k, arr in kwargs.items()}
        with torch.no_grad():
            return self.torch_model(*args, **kwargs).detach().numpy()
