from typing import Dict, Optional
import numpy

from coffea.ml_tools.helper import lazy_container, numpy_call_wrapper

try:
    import xgboost
except ImportError:
    warnings.warn(
        "Users should make sure the xgboost package is installed before proceeding!\n"
        "> pip install xgboost==1.5.1\n"
        "or\n"
        "> conda install xgboost==1.5.1",
        UserWarning,
    )
    raise err


class xgboost_wrapper(numpy_call_wrapper, lazy_container):
    """
    Very simple wrapper for xgbooster inference. The xgboost.Booster object is
    setup as a lazy object.
    """

    def __init__(self, fname):
        lazy_container.__init__(self, ["xgbooster"])
        self.xgboost_file = fname

    def _create_xgbooster(self) -> xgboost.Booster:
        # Automatic detection of compressed model file
        return xgboost.Booster(model_file=self.xgboost_file)

    def validate_numpy_input(
        self,
        data,
        dmat_args: Optional[Dict] = None,
        predict_args: Optional[Dict] = None,
    ):
        # TODO: Properly validate numpy data based on Booster values.
        pass

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
