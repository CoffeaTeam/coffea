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

from .helper import lazy_container, numpy_call_wrapper


class torch_wrapper(lazy_container, numpy_call_wrapper):
    """
    Wrapper for running pytorch with awkward/dask-awkward inputs.
    """

    def __init__(self, torch_model: torch.nn.Module):
        """
        Object lazy-ness is required to allow for GPU running on remote workers.
        The users will be responsible for passing in a properly constructed
        pytorch model (with dict_state properly loaded).

        Parameters
        ----------

        - torch_model: The torch model object that will be used for inference
        """
        lazy_container.__init__(self, ["model", "device"])

        # Reference to the original pytorch model, loading in the state, and
        # set to evaluation mode
        self.orig_model = torch_model

    def _create_device(self):
        """
        Torch device run calculations on. This wrapper class will always attempt
        to use GPU if possible. Setting this as a lazy object so that remote
        worker can have a different configuration the interactive session.
        """
        # Setting up the default torch inference computation device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_model(self):
        if torch.cuda.is_available():
            model = self.orig_model.cuda()
        else:
            model = self.orig_model
        model.eval()
        return model

    def validate_numpy_input(self, *args: numpy.array, **kwargs: numpy.array) -> None:
        # Pytorch's model.parameters is not a reliable way to extract input
        # information for arbitrary models, so we will leave this to the user.
        super().validate_numpy_input(*args, **kwargs)

    def numpy_call(self, *args: numpy.array, **kwargs: numpy.array) -> numpy.array:
        """
        Evaluating the numpy inputs via the model. Returning the results also as
        as numpy array.
        """
        args = [torch.from_numpy(arr).to(self.device) for arr in args]
        kwargs = {k: torch.from_numpy(arr).to(self.device) for k, arr in kwargs.items()}
        with torch.no_grad():
            return self.model(*args, **kwargs).detach().numpy()
