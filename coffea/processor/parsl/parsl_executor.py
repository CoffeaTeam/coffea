from coffea import hist, processor
from copy import deepcopy
from concurrent.futures import as_completed
from collections.abc import Sequence

from tqdm import tqdm
import cloudpickle as cpkl
import pickle as pkl
import lz4.frame as lz4f
import numpy as np
import time

from parsl.app.app import python_app
from .timeout import timeout
from ..executor import _futures_handler

lz4_clevel = 1


def coffea_pyapp_func(dataset, fn, treename, chunksize, index, procstr, timeout=None, flatten=True, **kwargs):
    raise RuntimeError('parsl_executor.coffea pyapp cannot be used any more,'
                       'please use a wrapped _work_function from processor.executor')


coffea_pyapp = timeout(python_app(coffea_pyapp_func))


class ParslExecutor(object):

    def __init__(self):
        self._counts = {}

    @property
    def counts(self):
        return self._counts

    def __call__(self, items, processor_instance, output, status=True, unit='items', desc='Processing', timeout=None, flatten=True, **kwargs):

        raise RuntimeError('ParslExecutor.__call__ cannot be used any more,'
                           'please use processor.parsl_executor')


parsl_executor = ParslExecutor()
