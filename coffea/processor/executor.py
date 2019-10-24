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

try:
    from collections.abc import Mapping, Sequence
except ImportError:
    from collections import Mapping, Sequence


_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


# instrument xrootd source
if not hasattr(uproot.source.xrootd.XRootDSource, '_read_real'):
    def _read(self, chunkindex):
        self.bytesread = getattr(self, 'bytesread', 0) + self._chunkbytes
        return self._read_real(chunkindex)

    uproot.source.xrootd.XRootDSource._read_real = uproot.source.xrootd.XRootDSource._read
    uproot.source.xrootd.XRootDSource._read = _read


class FileMeta(object):
    __slots__ = ['dataset', 'filename', 'treename', 'numentries']

    def __init__(self, dataset, filename, treename, numentries=None):
        self.dataset = dataset
        self.filename = filename
        self.treename = treename
        self.numentries = numentries

    def __hash__(self):
        # As used to lookup numentries, no need for dataset
        return hash((self.filename, self.treename))

    def __eq__(self, other):
        # In case of hash collisions
        return self.filename == other.filename and self.treename == other.treename

    def maybe_populate(self, cache):
        if cache and self in cache:
            self.numentries = cache[self]

    @property
    def populated(self):
        return self.numentries is not None

    def nchunks(self, target_chunksize):
        if not self.populated:
            raise RuntimeError
        return max(round(self.numentries / target_chunksize), 1)

    def chunks(self, target_chunksize):
        n = self.nchunks(target_chunksize)
        actual_chunksize = math.ceil(self.numentries / n)
        for index in range(n):
            yield WorkItem(self.dataset, self.filename, self.treename, actual_chunksize, index)


class WorkItem(object):
    __slots__ = ['dataset', 'filename', 'treename', 'chunksize', 'index']

    def __init__(self, dataset, filename, treename, chunksize, index):
        self.dataset = dataset
        self.filename = filename
        self.treename = treename
        self.chunksize = chunksize
        self.index = index


class _compression_wrapper(object):
    def __init__(self, level, function):
        self.level = level
        self.function = function

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


def _reduce(items):
    if len(items) == 0:
        raise ValueError("Empty list provided to reduction")
    # if dask has a cached result, we cannot alter it
    out = copy.deepcopy(_maybe_decompress(items.pop()))
    while items:
        out += _maybe_decompress(items.pop())
    return out


def _futures_handler(futures_set, output, status, unit, desc, add_fn):
    try:
        with tqdm(disable=not status, unit=unit, total=len(futures_set), desc=desc) as pbar:
            while len(futures_set) > 0:
                finished = set(job for job in futures_set if job.done())
                futures_set.difference_update(finished)
                while finished:
                    add_fn(output, finished.pop().result())
                    pbar.update(1)
                time.sleep(0.5)
    except KeyboardInterrupt:
        for job in futures_set:
            job.cancel()
        if status:
            print("Received SIGINT, killed pending jobs.  Running jobs will continue to completion.", file=sys.stderr)
            print("Running jobs:", sum(1 for j in futures_set if j.running()), file=sys.stderr)
    except Exception:
        for job in futures_set:
            job.cancel()
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
    """
    if len(items) == 0:
        return accumulator
    pool = kwargs.pop('pool', concurrent.futures.ProcessPoolExecutor)
    workers = kwargs.pop('workers', 1)
    status = kwargs.pop('status', True)
    unit = kwargs.pop('unit', 'items')
    desc = kwargs.pop('desc', 'Processing')
    clevel = kwargs.pop('compression', 1)
    if clevel is not None:
        function = _compression_wrapper(clevel, function)
    add_fn = _iadd
    if isinstance(pool, concurrent.futures.Executor):
        futures = set(pool.submit(function, item) for item in items)
        _futures_handler(futures, accumulator, status, unit, desc, add_fn)
    else:
        # assume its a class then
        with pool(max_workers=workers) as executor:
            futures = set(executor.submit(function, item) for item in items)
            _futures_handler(futures, accumulator, status, unit, desc, add_fn)
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
        heavy_input : serializable, optional
            Any value placed here will be broadcast to workers and joined to input
            items in a tuple (item, heavy_input) that is passed to function.
    """
    if len(items) == 0:
        return accumulator
    client = kwargs.pop('client')
    ntree = kwargs.pop('treereduction', 20)
    status = kwargs.pop('status', True)
    clevel = kwargs.pop('compression', 1)
    priority = kwargs.pop('priority', 0)
    heavy_input = kwargs.pop('heavy_input', None)
    reducer = _reduce
    if clevel is not None:
        function = _compression_wrapper(clevel, function)
        reducer = _compression_wrapper(clevel, reducer)

    if heavy_input is not None:
        heavy_token = client.scatter(heavy_input, broadcast=True, hash=False)
        items = list(zip(items, repeat(heavy_token)))
    futures = client.map(function, items, priority=priority)
    while len(futures) > 1:
        futures = client.map(
            reducer,
            [futures[i:i + ntree] for i in range(0, len(futures), ntree)],
            priority=priority,
        )
    if status:
        from dask.distributed import progress
        # FIXME: fancy widget doesn't appear, have to live with boring pbar
        progress(futures, multi=True, notebook=False)
    accumulator += _maybe_decompress(futures.pop().result())
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
    _futures_handler(futures, accumulator, status, unit, desc, add_fn)

    if cleanup:
        parsl.dfk().cleanup()
        parsl.clear()

    return accumulator


def _work_function(item, processor_instance, flatten=False, savemetrics=False, mmap=False):
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

    file = uproot.open(item.filename, localsource=localsource)
    tree = file[item.treename]
    df = LazyDataFrame(tree, item.chunksize, item.index, flatten=flatten)
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
    return wrapped_out


def _normalize_fileset(fileset, treename):
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


def _get_metadata(item):
    nentries = uproot.numentries(item.filename, item.treename)
    return set_accumulator([FileMeta(item.dataset, item.filename, item.treename, nentries)])


def run_uproot_job(fileset,
                   treename,
                   processor_instance,
                   executor,
                   executor_args={},
                   pre_executor=None,
                   pre_args=None,
                   chunksize=200000,
                   maxchunks=None,
                   metadata_cache=LRUCache(100000)
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
            'processor_compression' sets the compression level used to send processor instance
            to workers (default 1).
        pre_executor : callable
            A function like executor, used to calculate fileset metadata
            Defaults to executor
        pre_args : dict, optional
            Similar to executor_args, defaults to executor_args
        chunksize : int, optional
            Maximum number of entries to process at a time in the data frame.
        maxchunks : int, optional
            Maximum number of chunks to process per dataset
            Defaults to processing the whole dataset
        metadata_cache : mapping, optional
            A dict-like object to use as a cache for (file, tree) metadata that is used to
            determine chunking.  Defaults to a in-memory LRU cache that holds 100k entries
            (about 1MB depending on the length of filenames, etc.)  If you edit an input file
            (please don't) during a session, the session can be restarted to clear the cache.
    '''
    if not isinstance(fileset, Mapping):
        raise ValueError("Expected fileset to be a mapping dataset: list(files)")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")

    if pre_executor is None:
        pre_executor = executor
    if pre_args is None:
        pre_args = executor_args

    fileset = list(_normalize_fileset(fileset, treename))
    for filemeta in fileset:
        filemeta.maybe_populate(metadata_cache)

    chunks = []
    if maxchunks is None:
        # this is a bit of an abuse of map-reduce but ok
        to_get = set(filemeta for filemeta in fileset if not filemeta.populated)
        if len(to_get) > 0:
            out = set_accumulator()
            real_pre_args = {
                'desc': 'Preprocessing',
                'unit': 'file',
            }
            real_pre_args.update(pre_args)
            executor(to_get, _get_metadata, out, **real_pre_args)
            while out:
                item = out.pop()
                metadata_cache[item] = item.numentries
            for filemeta in fileset:
                filemeta.maybe_populate(metadata_cache)
        while fileset:
            filemeta = fileset.pop()
            for chunk in filemeta.chunks(chunksize):
                chunks.append(chunk)
    else:
        # get just enough file info to compute chunking
        nchunks = defaultdict(int)
        while fileset:
            filemeta = fileset.pop()
            if nchunks[filemeta.dataset] >= maxchunks:
                continue
            if not filemeta.populated:
                filemeta.numentries = _get_metadata(filemeta).pop().numentries
                metadata_cache[filemeta] = filemeta.numentries
            for chunk in filemeta.chunks(chunksize):
                chunks.append(chunk)
                nchunks[filemeta.dataset] += 1
                if nchunks[filemeta.dataset] >= maxchunks:
                    break

    # pop all _work_function args here
    savemetrics = executor_args.pop('savemetrics', False)
    flatten = executor_args.pop('flatten', False)
    mmap = executor_args.pop('mmap', False)
    pi_compression = executor_args.pop('processor_compression', 1)
    if pi_compression is None:
        pi_to_send = processor_instance
    else:
        pi_to_send = lz4f.compress(cloudpickle.dumps(processor_instance), compression_level=pi_compression)
    # hack around dask/dask#5503 which is really a silly request but here we are
    if executor is dask_executor:
        executor_args['heavy_input'] = pi_to_send
        closure = partial(_work_function,
                          processor_instance='heavy',
                          flatten=flatten,
                          savemetrics=savemetrics,
                          mmap=mmap,
                          )
    else:
        closure = partial(_work_function,
                          processor_instance=pi_to_send,
                          flatten=flatten,
                          savemetrics=savemetrics,
                          mmap=mmap,
                          )

    out = processor_instance.accumulator.identity()
    wrapped_out = dict_accumulator({'out': out, 'metrics': dict_accumulator()})
    exe_args = {
        'unit': 'chunk',
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
    executor_args.setdefault('flatten', True)
    executor_args.setdefault('compression', 0)

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
    executor_args.setdefault('cache', True)
    file_type = executor_args['file_type']
    treeName = executor_args['treeName']
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
    executor(spark, dfslist, processor_instance, output, thread_workers, use_cache)
    processor_instance.postprocess(output)

    if killSpark:
        _spark_stop(spark)
        del spark
        spark = None

    return output
