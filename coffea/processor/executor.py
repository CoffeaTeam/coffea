import concurrent.futures
from functools import partial
import time
import uproot
import pickle
from tqdm.auto import tqdm
import lz4.frame as lz4f
from . import ProcessorABC, LazyDataFrame
from .accumulator import AccumulatorABC, value_accumulator, set_accumulator, dict_accumulator

try:
    from collections.abc import Mapping, Sequence
    from functools import lru_cache
except ImportError:
    from collections import Mapping, Sequence

    def lru_cache(maxsize):
        def null_wrapper(f):
            return f
        return null_wrapper


# instrument xrootd source
if not hasattr(uproot.source.xrootd.XRootDSource, '_read_real'):
    def _read(self, chunkindex):
        self.bytesread = getattr(self, 'bytesread', 0) + self._chunkbytes
        return self._read_real(chunkindex)

    uproot.source.xrootd.XRootDSource._read_real = uproot.source.xrootd.XRootDSource._read
    uproot.source.xrootd.XRootDSource._read = _read


class _compression_wrapper(object):
    def __init__(self, level, function):
        self.level = level
        self.function = function

    # no @wraps due to pickle
    def __call__(self, *args, **kwargs):
        out = self.function(*args, **kwargs)
        return lz4f.compress(pickle.dumps(out), compression_level=self.level)


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
    # TODO: _maybe_decompress
    output += result


def _compressed_iadd(output, result):
    # TODO: deprecate in favor of smart _iadd
    print("decompress result %x" % id(result))
    output += pickle.loads(lz4f.decompress(result))
    print(output)


def _reduce(items):
    if len(items) == 0:
        raise ValueError("Empty list provided to reduction")
    out = _maybe_decompress(items.pop())
    while items:
        out += _maybe_decompress(items.pop())
    return out


def _futures_handler(futures_set, output, status, unit, desc, add_fn):
    try:
        with tqdm(disable=not status, unit=unit, total=len(futures_set), desc=desc) as pbar:
            while len(futures_set) > 0:
                finished = set(job for job in futures_set if job.done())
                futures_set.difference_update(finished)
                print("set", finished)
                while finished:
                    add_fn(output, finished.pop().result())
                    pbar.update(1)
                time.sleep(0.5)
    except KeyboardInterrupt:
        for job in futures_set:
            job.cancel()
        if status:
            print("Received SIGINT, killed pending jobs.  Running jobs will continue to completion.")
            print("Running jobs:", sum(1 for j in futures_set if j.running()))
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
        function : function
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
            Compress accumulator outputs in flight with LZ4, at level specified.
            Leave zero (default) for no compression.
    """
    status = kwargs.pop('status', True)
    unit = kwargs.pop('unit', 'items')
    desc = kwargs.pop('desc', 'Processing')
    clevel = kwargs.pop('compression', 0)
    if clevel > 0:
        function = _compression_wrapper(clevel, function)
    add_fn = _iadd if clevel == 0 else _compressed_iadd
    for i, item in tqdm(enumerate(items), disable=not status, unit=unit, total=len(items), desc=desc):
        add_fn(accumulator, function(item))
    return accumulator


def futures_executor(items, function, accumulator, **kwargs):
    """Execute using multiple local cores using python futures

    Parameters
    ----------
        items : list
            List of input arguments
        function : function
            A function to be called on each input, which returns an accumulator instance
        accumulator : AccumulatorABC
            An accumulator to collect the output of the function
        workers : int
            Number of parallel processes for futures
        status : bool
            If true (default), enable progress bar
        unit : str
            Label of progress bar unit
        desc : str
            Label of progress bar description
        compression : int, optional
            Compress accumulator outputs in flight with LZ4, at level specified.
            Leave zero (default) for no compression.
    """
    workers = kwargs.pop('workers', 1)
    status = kwargs.pop('status', True)
    unit = kwargs.pop('unit', 'items')
    desc = kwargs.pop('desc', 'Processing')
    clevel = kwargs.pop('compression', 0)
    if clevel > 0:
        function = _compression_wrapper(clevel, function)
    add_fn = _iadd if clevel == 0 else _compressed_iadd
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = set()
        futures.update(executor.submit(function, item) for item in items)
        _futures_handler(futures, accumulator, status, unit, desc, add_fn)
    return accumulator


def dask_executor(items, function, accumulator, **kwargs):
    """Execute using dask futures

    Parameters
    ----------
        items : list
            List of input arguments
        function : function
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
            Compress accumulator outputs in flight with LZ4, at level specified.
            Leave zero (default) for no compression.
    """
    client = kwargs.pop('client')
    ntree = kwargs.pop('treereduction', 20)
    status = kwargs.pop('status', True)
    clevel = kwargs.pop('compression', 0)
    reducer = _reduce
    if clevel > 0:
        function = _compression_wrapper(clevel, function)
        reducer = _compression_wrapper(clevel, reducer)

    futures = client.map(function, items)
    while len(futures) > 1:
        futures = client.map(
            reducer,
            [futures[i * ntree:(i + 1) * ntree] for i in range(len(futures) // ntree + 1)],
            priority=1,
        )
    if status:
        from dask.distributed import progress
        progress(futures, multi=False, notebook=False)
    accumulator += _maybe_decompress(futures.pop().result())
    return accumulator


def parsl_executor(items, function, accumulator, **kwargs):
    """Execute using parsl pyapp wrapper (TODO)

    Parameters
    ----------
        items : list
            List of input arguments
        function : function
            A function to be called on each input, which returns an accumulator instance
        accumulator : AccumulatorABC
            An accumulator to collect the output of the function
        config : parsl.config.Config, optional
            A parsl DataFlow configuration object. Necessary if there is no active kernel
        status : bool
            If true (default), enable progress bar
        unit : str
            Label of progress bar unit
        desc : str
            Label of progress bar description
        compression : int, optional
            Compress accumulator outputs in flight with LZ4, at level specified.
            Leave zero (default) for no compression.
    """
    import parsl
    from parsl.app.app import python_app
    from .parsl.timeout import timeout
    status = kwargs.pop('status', True)
    unit = kwargs.pop('unit', 'items')
    desc = kwargs.pop('desc', 'Processing')
    clevel = kwargs.pop('compression', 0)
    if clevel > 0:
        function = _compression_wrapper(clevel, function)
    add_fn = _iadd if clevel == 0 else _compressed_iadd

    cleanup = False
    if parsl.dfk() is None:
        config = kwargs.pop('config', None)
        if config is None:
            raise RuntimeError("No active parsl DataFlowKernel, must specify a config to construct one")
        parsl.clear()
        parsl.load(config)
        cleanup = True
    elif not isinstance(parsl.dfk(), parsl.dataflow.dflow.DataFlowKernel):
        raise RuntimeError("Expected parsl.dfk() to be a parsl.dataflow.dflow.DataFlowKernel")

    app = timeout(python_app(function))

    futures = set()
    futures.update(app(item) for item in items)
    _futures_handler(futures, accumulator, status, unit, desc, add_fn)

    if cleanup:
        parsl.dfk().cleanup()
        parsl.clear()

    return accumulator


def _work_function(item, processor_instance, flatten=False, savemetrics=False, mmap=False):
    dataset, filename, treename, chunksize, index = item
    if mmap:
        localsource = {}
    else:
        opts = dict(uproot.FileSource.defaults)
        opts.update({'parallel': None})

        def localsource(path):
            return uproot.FileSource(path, **opts)

    file = uproot.open(filename, localsource=localsource)
    tree = file[treename]
    df = LazyDataFrame(tree, chunksize, index, flatten=flatten)
    df['dataset'] = dataset
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


@lru_cache(maxsize=128)
def _get_chunking(filelist, treename, chunksize, workers=1):
    items = []
    executor = None if len(filelist) < 5 else concurrent.futures.ThreadPoolExecutor(workers)
    for filename, nentries in uproot.numentries(filelist, treename, total=False, executor=executor).items():
        for index in range(nentries // chunksize + 1):
            items.append((filename, chunksize, index))
    return items


def _get_chunking_lazy(filelist, treename, chunksize):
    for filename in filelist:
        nentries = uproot.numentries(filename, treename)
        for index in range(nentries // chunksize + 1):
            yield (filename, chunksize, index)


def run_uproot_job(fileset, treename, processor_instance, executor, executor_args={}, chunksize=500000, maxchunks=None):
    '''A tool to run jobs on a local machine

    A convenience wrapper to submit jobs for a file set, which is a
    dictionary of dataset: [file list] entries.  Supports only uproot
    reading, via the LazyDataFrame class.  For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.

    Parameters
    ----------
        fileset : dict
            dictionary {dataset: [file, file], }
        treename : str
            name of tree inside each root file, can be ``None``

            .. note:: treename can also be defined in fileset, which will override the passed treename
        processor_instance : ProcessorABC
            An instance of a class deriving from ProcessorABC
        executor : iterative_executor or futures_executor or dask_executor
            A function that takes 3 arguments: items, function, accumulator
            and performs some action equivalent to:
            ``for item in items: accumulator += function(item)``
        executor_args : dict, optional
            Extra arguments to pass to executor.  See `iterative_executor`,
            `futures_executor`, or `dask_executor` for available options.  Some options that
            affect the behavior of this function: 'pre_workers' specifies the
            number of parallel threads for calculating chunking (default: 4*workers);
            'savemetrics' saves some detailed metrics for xrootd processing (default False);
            'flatten' removes any jagged structure from the input files (default False).
        chunksize : int, optional
            Number of entries to process at a time in the data frame
        maxchunks : int, optional
            Maximum number of chunks to process per dataset
            Defaults to processing the whole dataset
    '''
    if not isinstance(fileset, Mapping):
        raise ValueError("Expected fileset to be a mapping dataset: list(files)")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")

    executor_args.setdefault('workers', 1)
    pre_workers = executor_args.pop('pre_workers', 4 * executor_args['workers'])

    local_treename = treename
    items = []
    for dataset, filelist in tqdm(fileset.items(), desc='Preprocessing'):
        if isinstance(filelist, dict):
            local_treename = filelist['treename'] if 'treename' in filelist else treename
            filelist = filelist['files']
        elif not isinstance(filelist, list):
            raise ValueError('list of filenames in fileset must be a list or a dict')

        if maxchunks is not None:
            chunks = _get_chunking_lazy(tuple(filelist), local_treename, chunksize)
        else:
            chunks = _get_chunking(tuple(filelist), local_treename, chunksize, pre_workers)
        for ichunk, chunk in enumerate(chunks):
            if (maxchunks is not None) and (ichunk > maxchunks):
                break
            items.append((dataset, chunk[0], local_treename, chunk[1], chunk[2]))

    # pop all _work_function args here
    savemetrics = executor_args.pop('savemetrics', False)
    flatten = executor_args.pop('flatten', False)
    mmap = executor_args.pop('mmap', False)
    closure = partial(_work_function,
                      processor_instance=processor_instance,
                      flatten=flatten,
                      savemetrics=savemetrics,
                      mmap=mmap,
                      )

    out = processor_instance.accumulator.identity()
    wrapped_out = dict_accumulator({'out': out, 'metrics': dict_accumulator()})
    executor(items, closure, wrapped_out, **executor_args)
    processor_instance.postprocess(out)
    if savemetrics:
        return out, wrapped_out['metrics']
    return out


def run_parsl_job(fileset, treename, processor_instance, executor, executor_args={}, chunksize=500000):
    '''A wrapper to submit parsl jobs

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
        print('you must have parsl installed to call run_parsl_job()!')
        raise e

    print('parsl version:', parsl.__version__)

    from .parsl.parsl_executor import ParslExecutor
    from .parsl.detail import _parsl_initialize, _parsl_stop, _parsl_get_chunking, _default_cfg

    if not isinstance(fileset, Mapping):
        raise ValueError("Expected fileset to be a mapping dataset: list(files)")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")
    if not isinstance(executor, ParslExecutor):
        raise ValueError("Expected executor to derive from ParslBaseExecutor")

    executor_args.setdefault('config', _default_cfg)
    executor_args.setdefault('timeout', 180)
    executor_args.setdefault('chunking_timeout', 10)
    executor_args.setdefault('flatten', True)

    # initialize parsl if we need to
    # if we initialize, then we deconstruct
    # when we're done
    killParsl = False
    if parsl.dfk() is None:
        _parsl_initialize(executor_args.get('config'))
        killParsl = True
    else:
        if not isinstance(parsl.dfk(), parsl.dataflow.dflow.DataFlowKernel):
            raise ValueError("Expected parsl.dfk() to be a parsl.dataflow.dflow.DataFlowKernel")

    tn = treename
    to_chunk = []
    for dataset, filelist in fileset.items():
        if isinstance(filelist, dict):
            tn = filelist['treename'] if 'treename' in filelist else tn
            filelist = filelist['files']
        if not isinstance(filelist, list):
            raise ValueError('list of filenames in fileset must be a list or a dict')
        if not isinstance(tn, str):
            tn = tuple(treename)
        for afile in filelist:
            to_chunk.append((dataset, afile, tn))

    items = _parsl_get_chunking(to_chunk, chunksize, timeout=executor_args['chunking_timeout'])

    # loose items we don't need any more
    executor_args.pop('chunking_timeout')
    executor_args.pop('config')

    output = processor_instance.accumulator.identity()
    executor(items, processor_instance, output, **executor_args)
    processor_instance.postprocess(output)

    if killParsl:
        _parsl_stop()

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
        print('you must have pyspark installed to call run_spark_job()!')
        raise e

    print('pyspark version:', pyspark.__version__)

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
    else:
        if not isinstance(spark, pyspark.sql.session.SparkSession):
            raise ValueError("Expected 'spark' to be a pyspark.sql.session.SparkSession")

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
