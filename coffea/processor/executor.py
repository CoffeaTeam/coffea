from __future__ import print_function, division
import concurrent.futures
from functools import partial
from itertools import repeat
import time
import uproot
import pickle
import sys
import math
import copy
import json
import cloudpickle
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
from ..util import _hash

try:
    from collections.abc import Mapping, Sequence
except ImportError:
    from collections import Mapping, Sequence


_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
DEFAULT_METADATA_CACHE = LRUCache(100000)


# instrument xrootd source
if not hasattr(uproot.source.xrootd.XRootDSource, '_read_real'):
    def _read(self, chunkindex):
        self.bytesread = getattr(self, 'bytesread', 0) + self._chunkbytes
        return self._read_real(chunkindex)

    uproot.source.xrootd.XRootDSource._read_real = uproot.source.xrootd.XRootDSource._read
    uproot.source.xrootd.XRootDSource._read = _read


class FileMeta(object):
    __slots__ = ['dataset', 'filename', 'treename', 'metadata']

    def __init__(self, dataset, filename, treename, metadata=None):
        self.dataset = dataset
        self.filename = filename
        self.treename = treename
        self.metadata = metadata

    def __hash__(self):
        # As used to lookup metadata, no need for dataset
        return _hash((self.filename, self.treename))

    def __eq__(self, other):
        # In case of hash collisions
        return self.filename == other.filename and self.treename == other.treename

    def maybe_populate(self, cache):
        if cache and self in cache:
            self.metadata = cache[self]

    def populated(self, clusters=False):
        '''Return true if metadata is populated

        By default, only require bare minimum metadata (numentries, uuid)
        If clusters is True, then require cluster metadata to be populated
        '''
        if self.metadata is None:
            return False
        elif clusters and 'clusters' not in self.metadata:
            return False
        return True

    def chunks(self, target_chunksize, align_clusters):
        if not self.populated(clusters=align_clusters):
            raise RuntimeError
        if align_clusters:
            chunks = [0]
            for c in self.metadata['clusters']:
                if c >= chunks[-1] + target_chunksize:
                    chunks.append(c)
            if self.metadata['clusters'][-1] != chunks[-1]:
                chunks.append(self.metadata['clusters'][-1])
            for start, stop in zip(chunks[:-1], chunks[1:]):
                yield WorkItem(self.dataset, self.filename, self.treename, start, stop, self.metadata['uuid'])
        else:
            n = max(round(self.metadata['numentries'] / target_chunksize), 1)
            actual_chunksize = math.ceil(self.metadata['numentries'] / n)
            for index in range(n):
                start, stop = actual_chunksize * index, min(self.metadata['numentries'], actual_chunksize * (index + 1))
                yield WorkItem(self.dataset, self.filename, self.treename, start, stop, self.metadata['uuid'])


class WorkItem(object):
    __slots__ = ['dataset', 'filename', 'treename', 'entrystart', 'entrystop', 'fileuuid']

    def __init__(self, dataset, filename, treename, entrystart, entrystop, fileuuid):
        self.dataset = dataset
        self.filename = filename
        self.treename = treename
        self.entrystart = entrystart
        self.entrystop = entrystop
        self.fileuuid = fileuuid


class _compression_wrapper(object):
    def __init__(self, level, function, name=None):
        self.level = level
        self.function = function
        self.name = name

    def __str__(self):
        if self.name is not None:
            return self.name
        try:
            name = self.function.__name__
            if name == "<lambda>":
                return "lambda"
            return name
        except AttributeError:
            return str(self.function)

    # no @wraps due to pickle
    def __call__(self, *args, **kwargs):
        out = self.function(*args, **kwargs)
        return lz4f.compress(pickle.dumps(out, protocol=_PICKLE_PROTOCOL), compression_level=self.level)


def _maybe_decompress(item):
    if isinstance(item, AccumulatorABC):
        return item
    try:
        item = pickle.loads(lz4f.decompress(item))
        if isinstance(item, AccumulatorABC):
            return item
        raise RuntimeError
    except (RuntimeError, pickle.UnpicklingError):
        raise ValueError("Executors can only reduce accumulators or LZ4-compressed pickled accumulators")


def _iadd(output, result):
    output += _maybe_decompress(result)


class _reduce(object):
    def __init__(self):
        pass

    def __str__(self):
        return "reduce"

    def __call__(self, items):
        if len(items) == 0:
            raise ValueError("Empty list provided to reduction")
        out = items.pop()
        if isinstance(out, AccumulatorABC):
            # if dask has a cached result, we cannot alter it, so make a copy
            out = copy.deepcopy(out)
        else:
            out = _maybe_decompress(out)
        while items:
            out += _maybe_decompress(items.pop())
        return out


def _cancel(job):
    try:
        # this is not implemented with parsl AppFutures
        job.cancel()
    except NotImplementedError:
        pass


def _futures_handler(futures_set, output, status, unit, desc, add_fn, tailtimeout):
    start = time.time()
    last_job = start
    try:
        with tqdm(disable=not status, unit=unit, total=len(futures_set), desc=desc) as pbar:
            while len(futures_set) > 0:
                finished = set(job for job in futures_set if job.done())
                futures_set.difference_update(finished)
                while finished:
                    add_fn(output, finished.pop().result())
                    pbar.update(1)
                    last_job = time.time()
                time.sleep(0.5)
                if tailtimeout is not None and (time.time() - last_job) > tailtimeout and (last_job - start) > 0:
                    njobs = len(futures_set)
                    for job in futures_set:
                        _cancel(job)
                        pbar.update(1)
                    import warnings
                    warnings.warn('Stopped {} jobs early due to tailtimeout = {}'.format(njobs, tailtimeout))
                    break
    except KeyboardInterrupt:
        for job in futures_set:
            _cancel(job)
        if status:
            print("Received SIGINT, killed pending jobs.  Running jobs will continue to completion.", file=sys.stderr)
            print("Running jobs:", sum(1 for j in futures_set if j.running()), file=sys.stderr)
    except Exception:
        for job in futures_set:
            _cancel(job)
        raise


def iterative_executor(items, function, accumulator, **kwargs):
    """Execute in one thread iteratively

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : AccumulatorABC
            An accumulator to collect the output of the function
        status : bool
            If true (default), enable progress bar
        unit : str
            Label of progress bar unit
        desc : str
            Label of progress bar description
        compression : int, optional
            Compress accumulator outputs in flight with LZ4, at level specified (default 1)
            Set to ``None`` for no compression.
    """
    if len(items) == 0:
        return accumulator
    status = kwargs.pop('status', True)
    unit = kwargs.pop('unit', 'items')
    desc = kwargs.pop('desc', 'Processing')
    clevel = kwargs.pop('compression', 1)
    if clevel is not None:
        function = _compression_wrapper(clevel, function)
    add_fn = _iadd
    for i, item in tqdm(enumerate(items), disable=not status, unit=unit, total=len(items), desc=desc):
        add_fn(accumulator, function(item))
    return accumulator


def futures_executor(items, function, accumulator, **kwargs):
    """Execute using multiple local cores using python futures

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : AccumulatorABC
            An accumulator to collect the output of the function
        pool : concurrent.futures.Executor class or instance, optional
            The type of futures executor to use, defaults to ProcessPoolExecutor.
            You can pass an instance instead of a class to re-use an executor
        workers : int, optional
            Number of parallel processes for futures (default 1)
        status : bool, optional
            If true (default), enable progress bar
        unit : str, optional
            Label of progress bar unit (default: 'Processing')
        desc : str, optional
            Label of progress bar description (default: 'items')
        compression : int, optional
            Compress accumulator outputs in flight with LZ4, at level specified (default 1)
            Set to ``None`` for no compression.
        tailtimeout : int, optional
            Timeout requirement on job tails. Cancel all remaining jobs if none have finished
            in the timeout window.
    """
    if len(items) == 0:
        return accumulator
    pool = kwargs.pop('pool', concurrent.futures.ProcessPoolExecutor)
    workers = kwargs.pop('workers', 1)
    status = kwargs.pop('status', True)
    unit = kwargs.pop('unit', 'items')
    desc = kwargs.pop('desc', 'Processing')
    clevel = kwargs.pop('compression', 1)
    tailtimeout = kwargs.pop('tailtimeout', None)
    if clevel is not None:
        function = _compression_wrapper(clevel, function)
    add_fn = _iadd
    if isinstance(pool, concurrent.futures.Executor):
        futures = set(pool.submit(function, item) for item in items)
        _futures_handler(futures, accumulator, status, unit, desc, add_fn, tailtimeout)
    else:
        # assume its a class then
        with pool(max_workers=workers) as executor:
            futures = set(executor.submit(function, item) for item in items)
            _futures_handler(futures, accumulator, status, unit, desc, add_fn, tailtimeout)
    return accumulator


def dask_executor(items, function, accumulator, **kwargs):
    """Execute using dask futures

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : AccumulatorABC
            An accumulator to collect the output of the function
        client : distributed.client.Client
            A dask distributed client instance
        treereduction : int, optional
            Tree reduction factor for output accumulators (default: 20)
        status : bool, optional
            If true (default), enable progress bar
        compression : int, optional
            Compress accumulator outputs in flight with LZ4, at level specified (default 1)
            Set to ``None`` for no compression.
        priority : int, optional
            Task priority, default 0
        retries : int, optional
            Number of retries for failed tasks (default: 3)
        heavy_input : serializable, optional
            Any value placed here will be broadcast to workers and joined to input
            items in a tuple (item, heavy_input) that is passed to function.
        function_name : str, optional
            Name of the function being passed

            .. note:: If ``heavy_input`` is set, ``function`` is assumed to be pure.
    """
    from dask.delayed import delayed
    if len(items) == 0:
        return accumulator
    client = kwargs.pop('client')
    ntree = kwargs.pop('treereduction', 20)
    status = kwargs.pop('status', True)
    clevel = kwargs.pop('compression', 1)
    priority = kwargs.pop('priority', 0)
    retries = kwargs.pop('retries', 3)
    heavy_input = kwargs.pop('heavy_input', None)
    function_name = kwargs.pop('function_name', None)
    reducer = _reduce()
    # secret options
    direct_heavy = kwargs.pop('direct_heavy', None)
    worker_affinity = kwargs.pop('worker_affinity', False)

    if clevel is not None:
        function = _compression_wrapper(clevel, function, name=function_name)
        reducer = _compression_wrapper(clevel, reducer)

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
        )
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
        # FIXME: fancy widget doesn't appear, have to live with boring pbar
        progress(work, multi=True, notebook=False)
    accumulator += _maybe_decompress(work.result())
    return accumulator


def parsl_executor(items, function, accumulator, **kwargs):
    """Execute using parsl pyapp wrapper

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : AccumulatorABC
            An accumulator to collect the output of the function
        config : parsl.config.Config, optional
            A parsl DataFlow configuration object. Necessary if there is no active kernel

            .. note:: In general, it is safer to construct the DFK with ``parsl.load(config)`` prior to calling this function
        status : bool
            If true (default), enable progress bar
        unit : str
            Label of progress bar unit
        desc : str
            Label of progress bar description
        compression : int, optional
            Compress accumulator outputs in flight with LZ4, at level specified (default 1)
            Set to ``None`` for no compression.
        tailtimeout : int, optional
            Timeout requirement on job tails. Cancel all remaining jobs if none have finished
            in the timeout window.
    """
    if len(items) == 0:
        return accumulator
    import parsl
    from parsl.app.app import python_app
    from .parsl.timeout import timeout
    status = kwargs.pop('status', True)
    unit = kwargs.pop('unit', 'items')
    desc = kwargs.pop('desc', 'Processing')
    clevel = kwargs.pop('compression', 1)
    tailtimeout = kwargs.pop('tailtimeout', None)
    if clevel is not None:
        function = _compression_wrapper(clevel, function)
    add_fn = _iadd

    cleanup = False
    config = kwargs.pop('config', None)
    try:
        parsl.dfk()
    except RuntimeError:
        cleanup = True
        pass
    if cleanup and config is None:
        raise RuntimeError("No active parsl DataFlowKernel, must specify a config to construct one")
    elif not cleanup and config is not None:
        raise RuntimeError("An active parsl DataFlowKernel already exists")
    elif config is not None:
        parsl.clear()
        parsl.load(config)

    app = timeout(python_app(function))

    futures = set(app(item) for item in items)
    _futures_handler(futures, accumulator, status, unit, desc, add_fn, tailtimeout)

    if cleanup:
        parsl.dfk().cleanup()
        parsl.clear()

    return accumulator


def _work_function(item, processor_instance, flatten=False, savemetrics=False,
                   mmap=False, nano=False, cachestrategy=None, skipbadfiles=False,
                   retries=0, xrootdtimeout=None):
    if processor_instance == 'heavy':
        item, processor_instance = item
    if not isinstance(processor_instance, ProcessorABC):
        processor_instance = cloudpickle.loads(lz4f.decompress(processor_instance))
    if mmap:
        localsource = {}
    else:
        opts = dict(uproot.FileSource.defaults)
        opts.update({'parallel': None})

        def localsource(path):
            return uproot.FileSource(path, **opts)

    import warnings
    out = processor_instance.accumulator.identity()
    retry_count = 0
    while retry_count <= retries:
        try:
            from uproot.source.xrootd import XRootDSource
            xrootdsource = XRootDSource.defaults
            xrootdsource['timeout'] = xrootdtimeout
            file = uproot.open(item.filename, localsource=localsource, xrootdsource=xrootdsource)
            if nano:
                cache = None
                if cachestrategy == 'dask-worker':
                    from distributed import get_worker
                    from .dask import ColumnCache
                    worker = get_worker()
                    try:
                        cache = worker.plugins[ColumnCache.name]
                    except KeyError:
                        # emit warning if not found?
                        pass
                df = NanoEvents.from_file(
                    file=file,
                    treename=item.treename,
                    entrystart=item.entrystart,
                    entrystop=item.entrystop,
                    metadata={'dataset': item.dataset},
                    cache=cache,
                )
            else:
                tree = file[item.treename]
                df = LazyDataFrame(tree, item.entrystart, item.entrystop, flatten=flatten)
                df['dataset'] = item.dataset
            tic = time.time()
            out = processor_instance.process(df)
            toc = time.time()
            metrics = dict_accumulator()
            if savemetrics:
                if isinstance(file.source, uproot.source.xrootd.XRootDSource):
                    metrics['bytesread'] = value_accumulator(int, file.source.bytesread)
                    metrics['dataservers'] = set_accumulator({file.source._source.get_property('DataServer')})
                metrics['columns'] = set_accumulator(df.materialized)
                metrics['entries'] = value_accumulator(int, df.size)
                metrics['processtime'] = value_accumulator(float, toc - tic)
            wrapped_out = dict_accumulator({'out': out, 'metrics': metrics})
            file.source.close()
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
            metrics = dict_accumulator()
            if savemetrics:
                metrics['bytesread'] = value_accumulator(int, 0)
                metrics['dataservers'] = set_accumulator({})
                metrics['columns'] = set_accumulator({})
                metrics['entries'] = value_accumulator(int, 0)
                metrics['processtime'] = value_accumulator(float, 0)
            wrapped_out = dict_accumulator({'out': out, 'metrics': metrics})
        except Exception as e:
            if retries == retry_count:
                raise e
            w_str = 'Attempt %d of %d. Will retry.' % (retry_count + 1, retries + 1)
            warnings.warn(w_str)
        retry_count += 1

    return wrapped_out


def _normalize_fileset(fileset, treename):
    if isinstance(fileset, str):
        with open(fileset) as fin:
            fileset = json.load(fin)
    for dataset, filelist in fileset.items():
        if isinstance(filelist, dict):
            local_treename = filelist['treename'] if 'treename' in filelist else treename
            filelist = filelist['files']
        elif isinstance(filelist, list):
            if treename is None:
                raise ValueError('treename must be specified if the fileset does not contain tree names')
            local_treename = treename
        else:
            raise ValueError('list of filenames in fileset must be a list or a dict')
        for filename in filelist:
            yield FileMeta(dataset, filename, local_treename)


def _get_metadata(item, skipbadfiles=False, retries=0, xrootdtimeout=None, align_clusters=False):
    import warnings
    out = set_accumulator()
    retry_count = 0
    while retry_count <= retries:
        try:
            # add timeout option according to modified uproot numentries defaults
            xrootdsource = {"timeout": xrootdtimeout, "chunkbytes": 32 * 1024, "limitbytes": 1024**2, "parallel": False}
            file = uproot.open(item.filename, xrootdsource=xrootdsource)
            tree = file[item.treename]
            metadata = {'numentries': tree.numentries, 'uuid': file._context.uuid}
            if align_clusters:
                metadata['clusters'] = [0] + list(c[1] for c in tree.clusters())
            out = set_accumulator([FileMeta(item.dataset, item.filename, item.treename, metadata)])
            break
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


def run_uproot_job(fileset,
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
    '''A tool to run a processor using uproot for data delivery

    A convenience wrapper to submit jobs for a file set, which is a
    dictionary of dataset: [file list] entries.  Supports only uproot
    reading, via the LazyDataFrame class.  For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.

    Parameters
    ----------
        fileset : dict
            A dictionary ``{dataset: [file, file], }``
            Optionally, if some files' tree name differ, the dictionary can be specified:
            ``{dataset: {'treename': 'name', 'files': [file, file]}, }``
        treename : str
            name of tree inside each root file, can be ``None``;
            treename can also be defined in fileset, which will override the passed treename
        processor_instance : ProcessorABC
            An instance of a class deriving from ProcessorABC
        executor : callable
            A function that takes 3 arguments: items, function, accumulator
            and performs some action equivalent to:
            ``for item in items: accumulator += function(item)``
        executor_args : dict, optional
            Arguments to pass to executor.  See `iterative_executor`,
            `futures_executor`, `dask_executor`, or `parsl_executor` for available options.
            Some options that affect the behavior of this function:
            'savemetrics' saves some detailed metrics for xrootd processing (default False);
            'flatten' removes any jagged structure from the input files (default False);
            'nano' builds the dataframe as a `NanoEvents` object rather than `LazyDataFrame` (default False);
            'processor_compression' sets the compression level used to send processor instance to workers (default 1).
            'skipbadfiles' instead of failing on a bad file, skip it (default False)
            'retries' optionally retry n times (default 0)
            'xrootdtimeout' timeout for xrootd read (seconds)
            'tailtimeout' timeout requirement on job tails (seconds)
            'align_clusters' aligns the chunks to natural boundaries in the ROOT files
        pre_executor : callable
            A function like executor, used to calculate fileset metadata
            Defaults to executor
        pre_args : dict, optional
            Similar to executor_args, defaults to executor_args
        chunksize : int, optional
            Maximum number of entries to process at a time in the data frame, default: 100k
        maxchunks : int, optional
            Maximum number of chunks to process per dataset
            Defaults to processing the whole dataset
        metadata_cache : mapping, optional
            A dict-like object to use as a cache for (file, tree) metadata that is used to
            determine chunking.  Defaults to a in-memory LRU cache that holds 100k entries
            (about 1MB depending on the length of filenames, etc.)  If you edit an input file
            (please don't) during a session, the session can be restarted to clear the cache.
    '''
    if not isinstance(fileset, (Mapping, str)):
        raise ValueError("Expected fileset to be a mapping dataset: list(files) or filename")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")

    if pre_executor is None:
        pre_executor = executor
    if pre_args is None:
        pre_args = dict(executor_args)
    if metadata_cache is None:
        metadata_cache = DEFAULT_METADATA_CACHE

    fileset = list(_normalize_fileset(fileset, treename))
    for filemeta in fileset:
        filemeta.maybe_populate(metadata_cache)

    # pop _get_metdata args here (also sent to _work_function)
    skipbadfiles = executor_args.pop('skipbadfiles', False)
    retries = executor_args.pop('retries', 0)
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
                'desc': 'Preprocessing',
                'unit': 'file',
                'compression': None,
                'tailtimeout': None,
                'worker_affinity': False,
            }
            pre_args.update(pre_arg_override)
            executor(to_get, metadata_fetcher, out, **pre_args)
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
            if not filemeta.populated(clusters=align_clusters):
                filemeta.metadata = metadata_fetcher(filemeta).pop().metadata
                metadata_cache[filemeta] = filemeta.metadata
            if skipbadfiles and not filemeta.populated(clusters=align_clusters):
                continue
            for chunk in filemeta.chunks(chunksize, align_clusters):
                chunks.append(chunk)
                nchunks[filemeta.dataset] += 1
                if nchunks[filemeta.dataset] >= maxchunks:
                    break

    # pop all _work_function args here
    savemetrics = executor_args.pop('savemetrics', False)
    flatten = executor_args.pop('flatten', False)
    mmap = executor_args.pop('mmap', False)
    nano = executor_args.pop('nano', False)
    cachestrategy = executor_args.pop('cachestrategy', None)
    pi_compression = executor_args.pop('processor_compression', 1)
    if pi_compression is None:
        pi_to_send = processor_instance
    else:
        pi_to_send = lz4f.compress(cloudpickle.dumps(processor_instance), compression_level=pi_compression)
    closure = partial(
        _work_function,
        flatten=flatten,
        savemetrics=savemetrics,
        mmap=mmap,
        nano=nano,
        cachestrategy=cachestrategy,
        skipbadfiles=skipbadfiles,
        retries=retries,
        xrootdtimeout=xrootdtimeout,
    )
    # hack around dask/dask#5503 which is really a silly request but here we are
    if executor is dask_executor:
        executor_args['heavy_input'] = pi_to_send
        closure = partial(closure, processor_instance='heavy')
    else:
        closure = partial(closure, processor_instance=pi_to_send)

    out = processor_instance.accumulator.identity()
    wrapped_out = dict_accumulator({'out': out, 'metrics': dict_accumulator()})
    exe_args = {
        'unit': 'chunk',
        'function_name': type(processor_instance).__name__,
    }
    exe_args.update(executor_args)
    executor(chunks, closure, wrapped_out, **exe_args)
    wrapped_out['metrics']['chunks'] = value_accumulator(int, len(chunks))
    processor_instance.postprocess(out)
    if savemetrics:
        return out, wrapped_out['metrics']
    return out


def run_parsl_job(fileset, treename, processor_instance, executor, executor_args={}, chunksize=200000):
    '''A wrapper to submit parsl jobs

    .. note:: Deprecated in favor of `run_uproot_job` with the `parsl_executor`

    Jobs are specified by a file set, which is a dictionary of
    dataset: [file list] entries.  Supports only uproot reading,
    via the LazyDataFrame class.  For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.

    Parameters
    ----------
        fileset : dict
            dictionary {dataset: [file, file], }
        treename : str
            name of tree inside each root file
        processor_instance : ProcessorABC
            An instance of a class deriving from ProcessorABC
        executor : coffea.processor.parsl.parsl_executor
            Must be the parsl executor, or otherwise derive from
            ``coffea.processor.parsl.ParslExecutor``
        executor_args : dict
            Extra arguments to pass to executor.  Special options
            interpreted here: 'config' provides a parsl dataflow
            configuration.
        chunksize : int, optional
            Number of entries to process at a time in the data frame

    '''

    try:
        import parsl
    except ImportError as e:
        print('you must have parsl installed to call run_parsl_job()!', file=sys.stderr)
        raise e

    import warnings

    warnings.warn("run_parsl_job is deprecated and will be removed in 0.7.0, replaced by run_uproot_job",
                  DeprecationWarning)

    from .parsl.parsl_executor import ParslExecutor
    from .parsl.detail import _default_cfg

    if not isinstance(fileset, Mapping):
        raise ValueError("Expected fileset to be a mapping dataset: list(files)")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")
    if isinstance(executor, ParslExecutor):
        warnings.warn("ParslExecutor class is deprecated replacing with processor.parsl_executor",
                      DeprecationWarning)
        executor = parsl_executor
    elif executor == parsl_executor:
        pass
    else:
        raise ValueError("Expected executor to derive from ParslExecutor or be executor.parsl_executor")

    executor_args.setdefault('config', _default_cfg)
    executor_args.setdefault('timeout', 180)
    executor_args.setdefault('chunking_timeout', 10)
    executor_args.setdefault('flatten', False)
    executor_args.setdefault('compression', 0)
    executor_args.setdefault('skipbadfiles', False)
    executor_args.setdefault('retries', 0)
    executor_args.setdefault('xrootdtimeout', None)

    try:
        parsl.dfk()
        executor_args.pop('config')
    except RuntimeError:
        pass

    output = run_uproot_job(fileset,
                            treename,
                            processor_instance=processor_instance,
                            executor=executor,
                            executor_args=executor_args)

    return output


def run_spark_job(fileset, processor_instance, executor, executor_args={},
                  spark=None, partitionsize=200000, thread_workers=16):
    '''A wrapper to submit spark jobs

    A convenience wrapper to submit jobs for spark datasets, which is a
    dictionary of dataset: [file list] entries.  Presently supports reading of
    parquet files converted from root.  For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.

    Parameters
    ----------
        fileset : dict
            dictionary {dataset: [file, file], }
        processor_instance : ProcessorABC
            An instance of a class deriving from ProcessorABC

            .. note:: The processor instance must define all the columns in data and MC that it reads as ``.columns``
        executor:
            anything that inherits from `SparkExecutor` like `spark_executor`

            In general, a function that takes 3 arguments: items, function accumulator
            and performs some action equivalent to:
            for item in items: accumulator += function(item)
        executor_args:
            arguments to send to the creation of a spark session
        spark:
            an optional already created spark instance

            if ``None`` then we create an ephemeral spark instance using a config
        partitionsize:
            partition size to try to aim for (coalescese only, repartition too expensive)
        thread_workers:
            how many spark jobs to let fly in parallel during processing steps
    '''

    try:
        import pyspark
    except ImportError as e:
        print('you must have pyspark installed to call run_spark_job()!', file=sys.stderr)
        raise e

    from packaging import version
    import pyarrow as pa
    import warnings
    arrow_env = ('ARROW_PRE_0_15_IPC_FORMAT', '1')
    if (version.parse(pa.__version__) >= version.parse('0.15.0') and
        version.parse(pyspark.__version__) < version.parse('3.0.0')):
        import os
        if (arrow_env[0] not in os.environ or
            os.environ[arrow_env[0]] != arrow_env[1]):
            warnings.warn('If you are using pyarrow >= 0.15.0, make sure to set %s=%s in your environment!' % arrow_env)

    import pyspark.sql
    from .spark.spark_executor import SparkExecutor
    from .spark.detail import _spark_initialize, _spark_stop, _spark_make_dfs

    if not isinstance(fileset, Mapping):
        raise ValueError("Expected fileset to be a mapping dataset: list(files)")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")
    if not isinstance(executor, SparkExecutor):
        raise ValueError("Expected executor to derive from SparkExecutor")

    executor_args.setdefault('config', None)
    executor_args.setdefault('file_type', 'parquet')
    executor_args.setdefault('laurelin_version', '0.3.0')
    executor_args.setdefault('treeName', 'Events')
    executor_args.setdefault('flatten', False)
    executor_args.setdefault('cache', True)
    executor_args.setdefault('skipbadfiles', False)
    executor_args.setdefault('retries', 0)
    executor_args.setdefault('xrootdtimeout', None)
    file_type = executor_args['file_type']
    treeName = executor_args['treeName']
    flatten = executor_args['flatten']
    use_cache = executor_args['cache']

    if executor_args['config'] is None:
        executor_args.pop('config')

    # initialize spark if we need to
    # if we initialize, then we deconstruct
    # when we're done
    killSpark = False
    if spark is None:
        spark = _spark_initialize(**executor_args)
        killSpark = True
        use_cache = False  # if we always kill spark then we cannot use the cache
    else:
        if not isinstance(spark, pyspark.sql.session.SparkSession):
            raise ValueError("Expected 'spark' to be a pyspark.sql.session.SparkSession")

    dfslist = {}
    if executor._cacheddfs is None:
        dfslist = _spark_make_dfs(spark, fileset, partitionsize, processor_instance.columns,
                                  thread_workers, file_type, treeName)

    output = processor_instance.accumulator.identity()
    executor(spark, dfslist, processor_instance, output, thread_workers, use_cache, flatten)
    processor_instance.postprocess(output)

    if killSpark:
        _spark_stop(spark)
        del spark
        spark = None

    return output
