import warnings
from typing import Dict, Optional

import numpy

from coffea.ml_tools.helper import nonserializable_attribute, numpy_call_wrapper

try:
    import xgboost
except ImportError as err:
    warnings.warn(
        "Users should make sure the xgboost package is installed before proceeding!\n"
        "> pip install xgboost==1.5.1\n"
        "or\n"
        "> conda install xgboost==1.5.1",
        UserWarning,
    )
    raise err


class xgboost_wrapper(numpy_call_wrapper, nonserializable_attribute):
    """
    Very simple wrapper for xgbooster inference. The xgboost.Booster object is
    nonserializable, so the users should pass in the xgboost model file.
    """

    def __init__(self, fname):
        nonserializable_attribute.__init__(self, ["xgbooster"])
        self.xgboost_file = fname

    def _create_xgbooster(self) -> xgboost.Booster:
        # Automatic detection of compressed model file
        return xgboost.Booster(model_file=self.xgboost_file)

    def validate_numpy_input(
        self,
        data: numpy.ndarray,
        dmat_args: Optional[Dict] = None,
        predict_args: Optional[Dict] = None,
    ):
        """
        The inner most dimension of the data array should be smaller than the
        number of features of the xgboost model. (Will raise a warning if
        mismatched). We will not attempt to parse the kwargs passed to the
        construction of a DMatrix, or the predict call, as those advanced
        features are expected to be properly handled by the user.
        """
        ndims = data.shape[-1]
        nfeat = self.xgbooster.num_features()
        if ndims > nfeat:
            raise ValueError(
                f"Input shape {data.shape} exceeded number of features ({nfeat})"
            )
        elif ndims < nfeat:
            warnings.warn(
                f"Input shape {data.shape} smaller than number of features ({nfeat})",
                UserWarning,
            )

    def numpy_call(
        self,
        data: numpy.ndarray,
        dmat_args: Optional[Dict] = None,
        predict_args: Optional[Dict] = None,
    ):
        """
        Passing the numpy array data as-is to the construction of an
        xgboost.DMatrix constructor (with additional keyword arguments should
        they be specified), the run the xgboost.Booster.predict method (with
        additional keyword arguments).
        """
        if dmat_args is None:
            dmat_args = {}
        if predict_args is None:
            predict_args = {}
        mat = xgboost.DMatrix(data, **dmat_args)
        return self.xgbooster.predict(mat, **predict_args)
