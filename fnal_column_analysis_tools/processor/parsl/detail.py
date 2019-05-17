from concurrent.futures import as_completed
import multiprocessing

import parsl

from parsl.app.app import python_app

from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

try:
    from functools import lru_cache
except ImportError:
    def lru_cache(maxsize):
        def null_wrapper(f):
            return f

        return null_wrapper

_default_cfg = Config(
    executors=[
        HighThroughputExecutor(
            label="coffea_parsl_default",
            cores_per_worker=1,
            provider=LocalProvider(
                channel=LocalChannel(),
                init_blocks=1,
                max_blocks=1,
            ),
        )
    ],
    strategy=None,
)


def _parsl_initialize(config=_default_cfg):
    dfk = parsl.load(config)
    return dfk


def _parsl_stop(dfk):
    dfk.cleanup()
    parsl.clear()


@python_app
def derive_chunks(filename, treename, chunksize):
    import uproot
    nentries = uproot.numentries(filename, treename)
    return [(filename, chunksize, index) for index in range(nentries // chunksize + 1)]


@lru_cache(maxsize=128)
def _parsl_get_chunking(filelist, treename, chunksize):
    fn_to_index = {fn: idx for idx, fn in enumerate(filelist)}
    future_to_fn = {derive_chunks(fn, treename, chunksize): fn for fn in filelist}

    temp = [0 for fn in filelist]
    for ftr in as_completed(future_to_fn):
        temp[fn_to_index[future_to_fn[ftr]]] = ftr.result()

    items = []
    for idx in range(len(temp)):
        items.extend(temp[idx])

    return items
