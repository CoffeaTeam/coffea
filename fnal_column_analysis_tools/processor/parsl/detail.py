from concurrent.futures import as_completed
import multiprocessing

import parsl

from parsl.app.app import python_app

from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

try:
    from collections.abc import Sequence
    from functools import lru_cache
except ImportError:
    from collections import Sequence

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
    from collections.abc import Sequence

    afile = uproot.open(filename)
    tree = None
    if isinstance(treename, str):
        tree = afile[treename]
    elif isinstance(treename, Sequence):
        for name in reversed(treename):
            if name in afile:
                tree = afile[name]
    else:
        raise Exception('treename must be a str or Sequence but is a %s!' % repr(type(treename)))

    if tree is None:
        raise Exception('No tree found, out of possible tree names: %s' % repr(treename))

    nentries = tree.numentries
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
