from concurrent.futures import as_completed

from parsl.app.app import python_app

from parsl.providers import LocalProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname

try:
    from functools import lru_cache
except ImportError:
    def lru_cache(maxsize):
        def null_wrapper(f):
            return f

        return null_wrapper


default_cfg = Config(
    executors=[
        HighThroughputExecutor(
            label="coffea_parsl_default",
            address=address_by_hostname(),
            prefetch_capacity=0,
            worker_debug=True,
            cores_per_worker=1,
            max_workers=1,
            # max_blocks=200,
            # workers_per_node=1,
            worker_logdir_root='./',
            provider=LocalProvider(
                channel=LocalChannel(),
                init_blocks=1,
                max_blocks=1,
                nodes_per_block=1
            ),
        )
    ],
    strategy=None,
)


def _parsl_work_function():
    raise NotImplementedError


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
