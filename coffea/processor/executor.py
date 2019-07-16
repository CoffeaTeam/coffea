import concurrent.futures
import time
from tqdm import tqdm
import uproot
from . import ProcessorABC, LazyDataFrame
from .accumulator import value_accumulator, set_accumulator, dict_accumulator

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


def iterative_executor(items, function, accumulator, status=True, unit='items', desc='Processing',
                       **kwargs):
    for i, item in tqdm(enumerate(items), disable=not status, unit=unit, total=len(items), desc=desc):
        accumulator += function(item, **kwargs)
    return accumulator


def default_future_add(output, result):
    output += result


def futures_handler(futures_set, output, status, unit, desc, futures_accumulator=default_future_add):
    try:
        with tqdm(disable=not status, unit=unit, total=len(futures_set), desc=desc) as pbar:
            while len(futures_set) > 0:
                finished = set(job for job in futures_set if job.done())
                for job in finished:
                    futures_accumulator(output, job.result())
                    pbar.update(1)
                job = None
                futures_set -= finished
                del finished
                time.sleep(1)
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


def futures_executor(items, function, accumulator, workers=1, status=True, unit='items', desc='Processing',
                     **kwargs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = set()
        futures.update(executor.submit(function, item, **kwargs) for item in items)
        futures_handler(futures, accumulator, status, unit, desc)
    return accumulator


def _work_function(item, flatten=False, savemetrics=False, mmap=False, **_):
    dataset, fn, treename, chunksize, index, processor_instance = item
    if mmap:
        localsource = {}
    else:
        opts = dict(uproot.FileSource.defaults)
        opts.update({'parallel': None})

        def localsource(path):
            return uproot.FileSource(path, **opts)

    file = uproot.open(fn, localsource=localsource)
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
    for fn, nentries in uproot.numentries(filelist, treename, total=False, executor=executor).items():
        for index in range(nentries // chunksize + 1):
            items.append((fn, chunksize, index))
    return items


def _get_chunking_lazy(filelist, treename, chunksize):
    for fn in filelist:
        nentries = uproot.numentries(fn, treename)
        for index in range(nentries // chunksize + 1):
            yield (fn, chunksize, index)


def run_uproot_job(fileset, treename, processor_instance, executor, executor_args={}, chunksize=500000, maxchunks=None):
    '''
    A convenience wrapper to submit jobs for a file set, which is a
    dictionary of dataset: [file list] entries.  Supports only uproot
    reading, via the LazyDataFrame class.  For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.

    Parameters
    ----------
        fileset:
            dictionary {dataset: [file, file], }
        treename:
            name of tree inside each root file, can be ``None``

            .. note:: treename can also be defined in fileset, which will override the passed treename
        processor_instance:
            an instance of a class deriving from ProcessorABC
        executor:
            any of `iterative_executor`, `futures_executor`, etc.

            In general, a function that takes 3 arguments: items, function accumulator
            and performs some action equivalent to:
            for item in items: accumulator += function(item)
        executor_args:
            extra arguments to pass to executor
            currently supported:
                workers: number of parallel processes for futures
                pre_workers: number of parallel threads for calculating chunking
                savemetrics: save some detailed metrics for xrootd processing
                flatten: flatten all branches returned by the dataframe (no jagged structure)
        chunksize:
            number of entries to process at a time in the data frame
        maxchunks:
            maximum number of chunks to process per dataset
    '''
    if not isinstance(fileset, Mapping):
        raise ValueError("Expected fileset to be a mapping dataset: list(files)")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")

    executor_args.setdefault('workers', 1)
    executor_args.setdefault('pre_workers', 4 * executor_args['workers'])
    executor_args.setdefault('savemetrics', False)

    tn = treename
    items = []
    for dataset, filelist in tqdm(fileset.items(), desc='Preprocessing'):
        if isinstance(filelist, dict):
            tn = filelist['treename'] if 'treename' in filelist else tn
            filelist = filelist['files']
        if not isinstance(filelist, list):
            raise ValueError('list of filenames in fileset must be a list or a dict')
        if maxchunks is not None:
            chunks = _get_chunking_lazy(tuple(filelist), tn, chunksize)
        else:
            chunks = _get_chunking(tuple(filelist), tn, chunksize, executor_args['pre_workers'])
        for ichunk, chunk in enumerate(chunks):
            if (maxchunks is not None) and (ichunk > maxchunks):
                break
            items.append((dataset, chunk[0], tn, chunk[1], chunk[2], processor_instance))

    out = processor_instance.accumulator.identity()
    wrapped_out = dict_accumulator({'out': out, 'metrics': dict_accumulator()})
    executor(items, _work_function, wrapped_out, **executor_args)
    processor_instance.postprocess(out)
    if executor_args['savemetrics']:
        return out, wrapped_out['metrics']
    return out


def run_parsl_job(fileset, treename, processor_instance, executor, data_flow=None, executor_args={}, chunksize=500000):
    '''
    A convenience wrapper to submit jobs for a file, which is a
    dictionary of dataset: [file list] entries. In this case using parsl.
    Supports only uproot reading, via the LazyDataFrame class.
    For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.

    Parameters
    ----------
        fileset:
            dictionary {dataset: [file, file], }
        treename:
            name of tree inside each root file
        processor_instance:
            an instance of a class deriving from ProcessorABC
        executor:
            anything that inherits from `ParslExecutor` like `parsl_executor`

            In general, a function that takes 3 arguments: items, function accumulator
            and performs some action equivalent to:
            for item in items: accumulator += function(item)
        executor_args:
            extra arguments to pass to executor
        chunksize:
            number of entries to process at a time in the data frame
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

    # initialize spark if we need to
    # if we initialize, then we deconstruct
    # when we're done
    killParsl = False
    if data_flow is None:
        data_flow = _parsl_initialize(**executor_args)
        killParsl = True
    else:
        if not isinstance(data_flow, parsl.dataflow.dflow.DataFlowKernel):
            raise ValueError("Expected 'data_flow' to be a parsl.dataflow.dflow.DataFlowKernel")

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
    executor(data_flow, items, processor_instance, output, **executor_args)
    processor_instance.postprocess(output)

    if killParsl:
        _parsl_stop(data_flow)

    return output


def run_spark_job(fileset, processor_instance, executor, executor_args={},
                  spark=None, partitionsize=200000, thread_workers=16):
    '''
    A convenience wrapper to submit jobs for spark datasets, which is a
    dictionary of dataset: [file list] entries.  Presently supports reading of
    parquet files converted from root.  For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.

    Parameters
    ----------
        fileset:
            dictionary {dataset: [file, file], }
        processor_instance:
            an instance of a class deriving from ProcessorABC

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
    executor_args.setdefault('laurelin_version', '0.1.0')
    file_type = executor_args['file_type']

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
                              thread_workers, file_type)

    output = processor_instance.accumulator.identity()
    executor(spark, dfslist, processor_instance, output, thread_workers)
    processor_instance.postprocess(output)

    if killSpark:
        _spark_stop(spark)
        del spark
        spark = None

    return output
