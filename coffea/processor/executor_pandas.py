from __future__ import print_function, division
import concurrent.futures
from functools import partial
from itertools import repeat
import time
import pickle
import sys
import math
import copy
import shutil
import json
import cloudpickle
import uproot4
import subprocess
import re
import os
from tqdm.auto import tqdm
from collections import defaultdict
from cachetools import LRUCache
import lz4.frame as lz4f
from .processor import ProcessorABC
from .accumulator import (
    AccumulatorABC,
    value_accumulator,
    set_accumulator,
    dict_accumulator,
)
from .dataframe import (
    LazyDataFrame,
)
from ..nanoaod import NanoEvents
from ..nanoevents import NanoEventsFactory, schemas
from ..util import _hash

try:
    from collections.abc import Mapping, Sequence
except ImportError:
    from collections import Mapping, Sequence


_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
DEFAULT_METADATA_CACHE = LRUCache(100000)

from .executor import _normalize_fileset, _get_metadata, Uproot3Context, _get_cache, _reduce, _hash
import pandas as pd
import dask.dataframe as dd

def _work_function(item, processor_instance, flatten=False,
                   mmap=False, schema=None, cachestrategy=None, skipbadfiles=False,
                   retries=0, xrootdtimeout=None):
    """
        Exactly the same as _work_function() from executor.py, except:
            - 'out' is a Pandas DataFrame, not AccumulatorABC
            - not saving any metrics, only the processor output
    """
    if processor_instance == 'heavy':
        item, processor_instance = item
    if not isinstance(processor_instance, ProcessorABC):
        processor_instance = cloudpickle.loads(lz4f.decompress(processor_instance))

    import warnings
    out = pd.DataFrame()
    retry_count = 0
    while retry_count <= retries:
        try:
            if schema is NanoEvents:
                # this is the only uproot3-dependent option
                filecontext = Uproot3Context(item.filename, xrootdtimeout, mmap)
            else:
                filecontext = uproot4.open(
                    item.filename,
                    timeout=xrootdtimeout,
                    file_handler=uproot4.MemmapSource if mmap else uproot4.MultithreadedFileSource,
                )
            with filecontext as file:
                if schema is None:
                    # To deprecate
                    tree = file[item.treename]
                    events = LazyDataFrame(tree, item.entrystart, item.entrystop, flatten=flatten)
                    events['dataset'] = item.dataset
                    events['filename'] = item.filename
                elif schema is NanoEvents:
                    # To deprecate
                    events = NanoEvents.from_file(
                        file=file,
                        treename=item.treename,
                        entrystart=item.entrystart,
                        entrystop=item.entrystop,
                        metadata={
                            'dataset': item.dataset,
                            'filename': item.filename,
                        },
                        cache=_get_cache(cachestrategy),
                    )
                elif issubclass(schema, schemas.BaseSchema):
                    materialized = []
                    factory = NanoEventsFactory.from_file(
                        file=file,
                        treepath=item.treename,
                        entry_start=item.entrystart,
                        entry_stop=item.entrystop,
                        runtime_cache=_get_cache(cachestrategy),
                        schemaclass=schema,
                        metadata={
                            'dataset': item.dataset,
                            'filename': item.filename,
                        },
                        access_log=materialized,
                    )
                    events = factory.events()
                else:
                    raise ValueError("Expected schema to derive from BaseSchema or NanoEvents, instead got %r" % schema)
                tic = time.time()
                try:
                    out = processor_instance.process(events)
                except Exception as e:
                    raise Exception(f"Failed processing file: {item.filename} ({item.entrystart}-{item.entrystop})") from e
                toc = time.time()
                return out
            break
        # catch xrootd errors and optionally skip
        # or retry to read the file
        except OSError as e:
            if not skipbadfiles:
                raise e
            else:
                w_str = 'Bad file source %s.' % item.filename
                if retries:
                    w_str += ' Attempt %d of %d.' % (retry_count + 1, retries + 1)
                    if retry_count + 1 < retries:
                        w_str += ' Will retry.'
                    else:
                        w_str += ' Skipping.'
                else:
                    w_str += ' Skipping.'
                warnings.warn(w_str)
        except Exception as e:
            if retries == retry_count:
                raise e
            w_str = 'Attempt %d of %d. Will retry.' % (retry_count + 1, retries + 1)
            warnings.warn(w_str)
        retry_count += 1
    return out


def dask_pandas_executor(items, function, accumulator, **kwargs):
    """
        Exactly the same as dask_executor except:
            - reducing is done only for 'get_metadata' function
            - output of the processor is collected into a distributed Dask DataFrame
            - Dask DataFrame is a Delayed object, actual output can be accessed by applying .compute()
    """
    if len(items) == 0:
        return accumulator
    client = kwargs.pop('client')
    ntree = kwargs.pop('treereduction', 20)
    status = kwargs.pop('status', True)
    priority = kwargs.pop('priority', 0)
    retries = kwargs.pop('retries', 3)
    heavy_input = kwargs.pop('heavy_input', None)
    function_name = kwargs.pop('function_name', None)
    reducer = _reduce()
    # secret options
    direct_heavy = kwargs.pop('direct_heavy', None)
    worker_affinity = kwargs.pop('worker_affinity', False)

    if heavy_input is not None:
        heavy_token = client.scatter(heavy_input, broadcast=True, hash=False, direct=direct_heavy)
        items = list(zip(items, repeat(heavy_token)))

    work = []
    if worker_affinity:
        workers = list(client.run(lambda: 0))

        def belongsto(workerindex, item):
            if heavy_input is not None:
                item = item[0]
            hashed = _hash((item.fileuuid, item.treename, item.entrystart, item.entrystop))
            return hashed % len(workers) == workerindex

        for workerindex, worker in enumerate(workers):
            work.extend(client.map(
                function,
                [item for item in items if belongsto(workerindex, item)],
                pure=(heavy_input is not None),
                priority=priority,
                retries=retries,
                workers={worker},
                allow_other_workers=False,
            ))
    else:
        work = client.map(
            function,
            items,
            pure=(heavy_input is not None),
            priority=priority,
            retries=retries,
            key=function_name,
        )

    # only reduce when getting the list of input files
    if (function_name=='get_metadata'):
        while len(work) > 1:
            work = client.map(
                reducer,
                [work[i:i + ntree] for i in range(0, len(work), ntree)],
                pure=True,
                priority=priority,
                retries=retries,
            )
        work = work[0]
        if status:
            from distributed import progress
            progress(work, multi=True, notebook=False)
        accumulator += work.result()
        return accumulator
    else:
        if status:
            from distributed import progress
            progress(work, multi=True, notebook=False)
        df = dd.from_delayed(work)
        accumulator['out'] = df
        return accumulator


def run_uproot_pandas_job(fileset,
                          treename,
                          processor_instance,
                          executor,
                          executor_args={},
                          pre_executor=None,
                          pre_args=None,
                          chunksize=100000,
                          maxchunks=None,
                          metadata_cache=None,
                          ):
    """ 
        Exactly the same as run_uproot_job() except:
            - output initialized as a Dask DataFrame
            - not saving metrics (there aren't any)
    """
    import warnings

    if not isinstance(fileset, (Mapping, str)):
        raise ValueError("Expected fileset to be a mapping dataset: list(files) or filename")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")

    # make a copy since we modify in-place
    executor_args = dict(executor_args)

    if pre_executor is None:
        pre_executor = executor
    if pre_args is None:
        pre_args = dict(executor_args)
    else:
        pre_args = dict(pre_args)
    if metadata_cache is None:
        metadata_cache = DEFAULT_METADATA_CACHE

    fileset = list(_normalize_fileset(fileset, treename))
    for filemeta in fileset:
        filemeta.maybe_populate(metadata_cache)

    # pop _get_metdata args here (also sent to _work_function)
    skipbadfiles = executor_args.pop('skipbadfiles', False)
    retries = 0
    xrootdtimeout = executor_args.pop('xrootdtimeout', None)
    align_clusters = executor_args.pop('align_clusters', False)
    metadata_fetcher = partial(_get_metadata,
                               skipbadfiles=skipbadfiles,
                               retries=retries,
                               xrootdtimeout=xrootdtimeout,
                               align_clusters=align_clusters,
                               )

    chunks = []
    if maxchunks is None:
        # this is a bit of an abuse of map-reduce but ok
        to_get = set(filemeta for filemeta in fileset if not filemeta.populated(clusters=align_clusters))
        if len(to_get) > 0:
            out = set_accumulator()
            pre_arg_override = {
                'function_name': 'get_metadata',
                'desc': 'Preprocessing',
                'unit': 'file',
                'compression': None,
                'tailtimeout': None,
                'worker_affinity': False,
            }
            pre_args.update(pre_arg_override)
            pre_executor(to_get, metadata_fetcher, out, **pre_args)
            while out:
                item = out.pop()
                metadata_cache[item] = item.metadata
            for filemeta in fileset:
                filemeta.maybe_populate(metadata_cache)
        while fileset:
            filemeta = fileset.pop()
            if skipbadfiles and not filemeta.populated(clusters=align_clusters):
                continue
            for chunk in filemeta.chunks(chunksize, align_clusters):
                chunks.append(chunk)
    else:
        # get just enough file info to compute chunking
        nchunks = defaultdict(int)
        while fileset:
            filemeta = fileset.pop()
            if nchunks[filemeta.dataset] >= maxchunks:
                continue
            if skipbadfiles and not filemeta.populated(clusters=align_clusters):
                continue
            if not filemeta.populated(clusters=align_clusters):
                filemeta.metadata = metadata_fetcher(filemeta).pop().metadata
                metadata_cache[filemeta] = filemeta.metadata
            for chunk in filemeta.chunks(chunksize, align_clusters):
                chunks.append(chunk)
                nchunks[filemeta.dataset] += 1
                if nchunks[filemeta.dataset] >= maxchunks:
                    break

    # pop all _work_function args here
    if "flatten" in executor_args:
        warnings.warn("Executor argument 'flatten' is deprecated, please refactor your processor to accept awkward arrays", DeprecationWarning)
    flatten = executor_args.pop('flatten', False)
    mmap = executor_args.pop('mmap', False)
    schema = executor_args.pop('schema', None)
    nano = executor_args.pop('nano', False)
    if nano:
        warnings.warn("Please use 'schema': processor.NanoEvents rather than 'nano': True to enable awkward0 NanoEvents processing", DeprecationWarning)
        schema = NanoEvents
    cachestrategy = executor_args.pop('cachestrategy', None)
    pi_compression = executor_args.pop('processor_compression', 1)
    if pi_compression is None:
        pi_to_send = processor_instance
    else:
        pi_to_send = lz4f.compress(cloudpickle.dumps(processor_instance), compression_level=pi_compression)
    closure = partial(
        _work_function,
        flatten=flatten,
        mmap=mmap,
        schema=schema,
        cachestrategy=cachestrategy,
        skipbadfiles=skipbadfiles,
        retries=retries,
        xrootdtimeout=xrootdtimeout,
    )
    # hack around dask/dask#5503 which is really a silly request but here we are
    executor_args['heavy_input'] = pi_to_send
    closure = partial(closure, processor_instance='heavy')

    out = dd.from_pandas(pd.DataFrame(), npartitions=1)
    wrapped_out = dict_accumulator({'out': out})
    exe_args = {
        'unit': 'chunk',
        'function_name': type(processor_instance).__name__,
    }
    exe_args.update(executor_args)
    executor(chunks, closure, wrapped_out, **exe_args)
    processor_instance.postprocess(out)

    return wrapped_out['out']