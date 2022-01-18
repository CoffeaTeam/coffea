from __future__ import print_function, division
import concurrent.futures
from functools import partial
from itertools import repeat
import time
import pickle
import sys
import math
import json
import cloudpickle
import uproot
import uuid
import warnings
import shutil
from tqdm.auto import tqdm
from collections import defaultdict
from cachetools import LRUCache
import lz4.frame as lz4f
from .processor import ProcessorABC
from .accumulator import accumulate, set_accumulator, Accumulatable
from .dataframe import LazyDataFrame
from ..nanoevents import NanoEventsFactory, schemas
from ..util import _hash

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, asdict
from typing import Iterable, Callable, Optional, List, Generator, Dict, Union


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


try:
    from functools import cached_property
except ImportError:
    cached_property = property


_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
DEFAULT_METADATA_CACHE: MutableMapping = LRUCache(100000)

_PROTECTED_NAMES = {
    "dataset",
    "filename",
    "treename",
    "metadata",
    "entrystart",
    "entrystop",
    "fileuuid",
    "numentries",
    "uuid",
    "clusters",
}


class FileMeta(object):
    __slots__ = ["dataset", "filename", "treename", "metadata"]

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
        """Return true if metadata is populated

        By default, only require bare minimum metadata (numentries, uuid)
        If clusters is True, then require cluster metadata to be populated
        """
        if self.metadata is None:
            return False
        elif "numentries" not in self.metadata or "uuid" not in self.metadata:
            return False
        elif clusters and "clusters" not in self.metadata:
            return False
        return True

    def chunks(self, target_chunksize, align_clusters, dynamic_chunksize):
        if align_clusters and dynamic_chunksize:
            raise RuntimeError(
                "align_clusters cannot be used with a dynamic chunksize."
            )
        if not self.populated(clusters=align_clusters):
            raise RuntimeError
        user_keys = set(self.metadata.keys()) - _PROTECTED_NAMES
        user_meta = {k: self.metadata[k] for k in user_keys}
        if align_clusters:
            chunks = [0]
            for c in self.metadata["clusters"]:
                if c >= chunks[-1] + target_chunksize:
                    chunks.append(c)
            if self.metadata["clusters"][-1] != chunks[-1]:
                chunks.append(self.metadata["clusters"][-1])
            for start, stop in zip(chunks[:-1], chunks[1:]):
                yield WorkItem(
                    self.dataset,
                    self.filename,
                    self.treename,
                    start,
                    stop,
                    self.metadata["uuid"],
                    user_meta,
                )
            return target_chunksize
        else:
            n = max(round(self.metadata["numentries"] / target_chunksize), 1)
            actual_chunksize = math.ceil(self.metadata["numentries"] / n)

            start = 0
            while start < self.metadata["numentries"]:
                stop = min(self.metadata["numentries"], start + actual_chunksize)
                next_chunksize = yield WorkItem(
                    self.dataset,
                    self.filename,
                    self.treename,
                    start,
                    stop,
                    self.metadata["uuid"],
                    user_meta,
                )
                start = stop
                if dynamic_chunksize and next_chunksize:
                    n = max(
                        math.ceil(
                            (self.metadata["numentries"] - start) / next_chunksize
                        ),
                        1,
                    )
                    actual_chunksize = math.ceil(
                        (self.metadata["numentries"] - start) / n
                    )
            if dynamic_chunksize and next_chunksize:
                return next_chunksize
            else:
                return target_chunksize


@dataclass(unsafe_hash=True)
class WorkItem:
    dataset: str
    filename: str
    treename: str
    entrystart: int
    entrystop: int
    fileuuid: str
    usermeta: Optional[Dict] = field(default=None, compare=False)

    def __len__(self) -> int:
        return self.entrystop - self.entrystart


def _compress(item, compression):
    return lz4f.compress(
        pickle.dumps(item, protocol=_PICKLE_PROTOCOL), compression_level=compression
    )


def _decompress(item):
    return pickle.loads(lz4f.decompress(item))


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
        return _compress(out, self.level)


class _reduce:
    def __init__(self, compression):
        self.compression = compression

    def __str__(self):
        return "reduce"

    def __call__(self, items):
        items = list(items)
        if len(items) == 0:
            raise ValueError("Empty list provided to reduction")
        if self.compression is not None:
            out = _decompress(items.pop())
            out = accumulate(map(_decompress, items), out)
            return _compress(out, self.compression)
        return accumulate(items)


def _futures_handler(futures, timeout):
    """Essentially the same as concurrent.futures.as_completed
    but makes sure not to hold references to futures any longer than strictly necessary,
    which is important if the future holds a large result.
    """
    futures = set(futures)
    try:
        while futures:
            try:
                done, futures = concurrent.futures.wait(
                    futures,
                    timeout=timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if len(done) == 0:
                    warnings.warn(
                        f"No finished jobs after {timeout}s, stopping remaining {len(futures)} jobs early"
                    )
                    break
                while done:
                    try:
                        yield done.pop().result()
                    except concurrent.futures.CancelledError:
                        pass
            except KeyboardInterrupt as e:
                for job in futures:
                    try:
                        job.cancel()
                        # this is not implemented with parsl AppFutures
                    except NotImplementedError:
                        raise e from None
                running = sum(job.running() for job in futures)
                warnings.warn(
                    f"Early stop: cancelled {len(futures) - running} jobs, will wait for {running} running jobs to complete"
                )
    finally:
        running = sum(job.running() for job in futures)
        if running:
            warnings.warn(
                f"Cancelling {running} running jobs (likely due to an exception)"
            )
        try:
            while futures:
                futures.pop().cancel()
        except NotImplementedError:
            pass


@dataclass
class ExecutorBase:
    # shared by all executors
    status: bool = True
    unit: str = "items"
    desc: str = "Processing"
    compression: Optional[int] = 1
    function_name: Optional[str] = None

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        raise NotImplementedError(
            "This class serves as a base class for executors, do not instantiate it!"
        )

    def copy(self, **kwargs):
        tmp = self.__dict__.copy()
        tmp.update(kwargs)
        return type(self)(**tmp)


@dataclass
class WorkQueueExecutor(ExecutorBase):
    """Execute using Work Queue

    For more information, see :ref:`intro-coffea-wq`

    Parameters
    ----------
        items : list or generator
            Sequence of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : Accumulatable
            An accumulator to collect the output of the function
        status : bool
            If true (default), enable progress bar
        unit : str
            Label of progress bar unit
        desc : str
            Label of progress bar description
        compression : int, optional
            Compress accumulator outputs in flight with LZ4, at level specified (default 9)
            Set to ``None`` for no compression.
        # work queue specific options:
        cores : int
            Number of cores for work queue task. If unset, use a whole worker.
        memory : int
            Amount of memory (in MB) for work queue task. If unset, use a whole worker.
        disk : int
            Amount of disk space (in MB) for work queue task. If unset, use a whole worker.
        gpus : int
            Number of GPUs to allocate to each task.  If unset, use zero.
        resource_monitor : str
            If given, one of 'off', 'measure', or 'watchdog'. Default is 'off'.
            - 'off': turns off resource monitoring. Overriden if resources_mode
                     is not set to 'fixed'.
            - 'measure': turns on resource monitoring for Work Queue. The
                        resources used per task are measured.
            - 'watchdog': in addition to measuring resources, tasks are terminated if they
                        go above the cores, memory, or disk specified.
        resources_mode : str
            one of 'fixed', 'max-seen', or 'max-throughput'. Default is 'fixed'.
            Sets the strategy to automatically allocate resources to tasks.
            - 'fixed': allocate cores, memory, and disk specified for each task.
            - 'max-seen' or 'auto': use the cores, memory, and disk given as maximum values to allocate,
                          but first try each task by allocating the maximum values seen. Leads
                          to a good compromise between parallelism and number of retries.
            - 'max-throughput': Like max-seen, but first tries the task with an
                          allocation that maximizes overall throughput.
            If resources_mode is other than 'fixed', preprocessing and
            accumulation tasks always use the 'max-seen' strategy, as the
            former tasks always use the same resources, the latter has a
            distribution of resources that increases over time.
        split_on_exhaustion: bool
            Whether to split a processing task in half according to its chunksize when it exhausts its
            the cores, memory, or disk allocated to it. If False, a task that exhausts resources
            permanently fails. Default is True.
        fast_terminate_workers: int
            Terminate workers on which tasks have been running longer than average.
            The time limit is computed by multiplying the average runtime of tasks
            by the value of 'fast_terminate_workers'. Since there are
            legitimately slow tasks, no task may trigger fast termination in
            two distinct workers. Less than 1 disables it.

        master_name : str
            Name to refer to this work queue master.
            Sets port to 0 (any available port) if port not given.
        port : int
            Port number for work queue master program. Defaults to 9123 if
            master_name not given.
        password_file: str
            Location of a file containing a password used to authenticate workers.

        extra_input_files: list
            A list of files in the current working directory to send along with each task.
            Useful for small custom libraries and configuration files needed by the processor.
        x509_proxy : str
            Path to the X509 user proxy. If None (the default), use the value of the
            environment variable X509_USER_PROXY, or fallback to the file /tmp/x509up_u${UID} if
            exists.  If False, disables the default behavior and no proxy is sent.

        environment_file : optional, str
            Conda python environment tarball to use. If not given, assume that
            the python environment is already setup at the execution site.
        wrapper : str
            Wrapper script to run/open python environment tarball. Defaults to python_package_run found in PATH.

        chunks_per_accum : int
            Number of processed chunks per accumulation task. Defaults is 10.
        chunks_accum_in_mem : int
            Maximum number of chunks to keep in memory at each accumulation step in an accumulation task. Default is 2.

        verbose : bool
            If true, emit a message on each task submission and completion.
            Default is false.
        debug_log : str
            Filename for debug output
        stats_log : str
            Filename for tasks statistics output
        transactions_log : str
            Filename for tasks lifetime reports output
        print_stdout : bool
            If true (default), print the standard output of work queue task on completion.

        custom_init : function, optional
            A function that takes as an argument the queue's WorkQueue object.
            The function is called just before the first work unit is submitted
            to the queue.
    """

    # Standard executor options:
    compression: Optional[int] = 9  # as recommended by lz4
    retries: int = 2  # task executes at most 3 times
    # wq executor options:
    master_name: Optional[str] = None
    port: Optional[int] = None
    filepath: str = "."
    events_total: Optional[int] = None
    x509_proxy: Optional[str] = None
    verbose: bool = False
    print_stdout: bool = False
    bar_format: str = "{desc:<14}{percentage:3.0f}%|{bar}{r_bar:<55}"
    debug_log: Optional[str] = None
    stats_log: Optional[str] = None
    transactions_log: Optional[str] = None
    password_file: Optional[str] = None
    environment_file: Optional[str] = None
    extra_input_files: List = field(default_factory=list)
    wrapper: Optional[str] = shutil.which("python_package_run")
    resource_monitor: Optional[str] = "off"
    resources_mode: Optional[str] = "fixed"
    split_on_exhaustion: Optional[bool] = True
    fast_terminate_workers: Optional[int] = None
    cores: Optional[int] = None
    memory: Optional[int] = None
    disk: Optional[int] = None
    gpus: Optional[int] = None
    chunks_per_accum: int = 10
    chunks_accum_in_mem: int = 2
    chunksize: int = 1024
    dynamic_chunksize: Optional[Dict] = None
    custom_init: Optional[Callable] = None

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        try:
            import work_queue  # noqa
            import dill  # noqa
            from .work_queue_tools import work_queue_main
        except ImportError as e:
            print(
                "You must have Work Queue and dill installed to use WorkQueueExecutor!"
            )
            raise e

        from .work_queue_tools import _get_x509_proxy

        if self.x509_proxy is None:
            self.x509_proxy = _get_x509_proxy()

        return work_queue_main(
            items,
            function,
            accumulator,
            **self.__dict__,
        )


@dataclass
class IterativeExecutor(ExecutorBase):
    """Execute in one thread iteratively

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : Accumulatable
            An accumulator to collect the output of the function
        status : bool
            If true (default), enable progress bar
        unit : str
            Label of progress bar unit
        desc : str
            Label of progress bar description
        compression : int, optional
            Ignored for iterative executor
    """

    workers: int = 1

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        if len(items) == 0:
            return accumulator
        gen = tqdm(
            items,
            disable=not self.status,
            unit=self.unit,
            total=len(items),
            desc=self.desc,
        )
        gen = map(function, gen)
        return accumulate(gen, accumulator)


@dataclass
class FuturesExecutor(ExecutorBase):
    """Execute using multiple local cores using python futures

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : Accumulatable
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

    pool: Union[Callable[..., concurrent.futures.Executor], concurrent.futures.Executor] = concurrent.futures.ProcessPoolExecutor  # fmt: skip
    workers: int = 1
    tailtimeout: Optional[int] = None

    def __getstate__(self):
        return dict(self.__dict__, pool=None)

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        if len(items) == 0:
            return accumulator
        if self.compression is not None:
            function = _compression_wrapper(self.compression, function)

        def processwith(pool):
            gen = _futures_handler(
                {pool.submit(function, item) for item in items}, self.tailtimeout
            )
            try:
                return accumulate(
                    tqdm(
                        gen if self.compression is None else map(_decompress, gen),
                        disable=not self.status,
                        unit=self.unit,
                        total=len(items),
                        desc=self.desc,
                    ),
                    accumulator,
                )
            finally:
                gen.close()

        if isinstance(self.pool, concurrent.futures.Executor):
            return processwith(pool=self.pool)
        else:
            # assume its a class then
            with self.pool(max_workers=self.workers) as poolinstance:
                return processwith(pool=poolinstance)


@dataclass
class DaskExecutor(ExecutorBase):
    """Execute using dask futures

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : Accumulatable
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
        use_dataframes: bool, optional
            Retrieve output as a distributed Dask DataFrame (default: False).
            The outputs of individual tasks must be Pandas DataFrames.

            .. note:: If ``heavy_input`` is set, ``function`` is assumed to be pure.
    """

    client: Optional["dask.distributed.Client"] = None  # noqa
    treereduction: int = 20
    priority: int = 0
    retries: int = 3
    heavy_input: Optional[bytes] = None
    use_dataframes: bool = False
    # secret options
    worker_affinity: bool = False

    def __getstate__(self):
        return dict(self.__dict__, client=None)

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        if len(items) == 0:
            return accumulator

        import dask.dataframe as dd
        from dask.distributed import Client
        from distributed.scheduler import KilledWorker

        if self.client is None:
            self.client = Client(threads_per_worker=1)

        if self.use_dataframes:
            self.compression = None

        reducer = _reduce(self.compression)
        if self.compression is not None:
            function = _compression_wrapper(
                self.compression, function, name=self.function_name
            )

        if self.heavy_input is not None:
            # client.scatter is not robust against adaptive clusters
            # https://github.com/CoffeaTeam/coffea/issues/465
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Large object of size")
                items = list(
                    zip(
                        items, repeat(self.client.submit(lambda x: x, self.heavy_input))
                    )
                )

        work = []
        key_to_item = {}
        if self.worker_affinity:
            workers = list(self.client.run(lambda: 0))

            def belongsto(heavy_input, workerindex, item):
                if heavy_input is not None:
                    item = item[0]
                hashed = _hash(
                    (item.fileuuid, item.treename, item.entrystart, item.entrystop)
                )
                return hashed % len(workers) == workerindex

            for workerindex, worker in enumerate(workers):
                items_worker = [
                    item
                    for item in items
                    if belongsto(self.heavy_input, workerindex, item)
                ]
                work_worker = self.client.map(
                    function,
                    items_worker,
                    pure=(self.heavy_input is not None),
                    priority=self.priority,
                    retries=self.retries,
                    workers={worker},
                    allow_other_workers=False,
                )
                work.extend(work_worker)
                key_to_item.update(
                    {
                        future.key: item
                        for future, item in zip(work_worker, items_worker)
                    }
                )
        else:
            work = self.client.map(
                function,
                items,
                pure=(self.heavy_input is not None),
                priority=self.priority,
                retries=self.retries,
            )
            key_to_item.update({future.key: item for future, item in zip(work, items)})
        if (self.function_name == "get_metadata") or not self.use_dataframes:
            while len(work) > 1:
                work = self.client.map(
                    reducer,
                    [
                        work[i : i + self.treereduction]
                        for i in range(0, len(work), self.treereduction)
                    ],
                    pure=True,
                    priority=self.priority,
                    retries=self.retries,
                )
                key_to_item.update({future.key: "(output reducer)" for future in work})
            work = work[0]
            try:
                if self.status:
                    from distributed import progress

                    # FIXME: fancy widget doesn't appear, have to live with boring pbar
                    progress(work, multi=True, notebook=False)
                return accumulate(
                    [
                        work.result()
                        if self.compression is None
                        else _decompress(work.result())
                    ],
                    accumulator,
                )
            except KilledWorker as ex:
                baditem = key_to_item[ex.task]
                if self.heavy_input is not None and isinstance(baditem, tuple):
                    baditem = baditem[0]
                raise RuntimeError(
                    f"Work item {baditem} caused a KilledWorker exception (likely a segfault or out-of-memory issue)"
                )
        else:
            if self.status:
                from distributed import progress

                progress(work, multi=True, notebook=False)
            return {"out": dd.from_delayed(work)}


@dataclass
class ParslExecutor(ExecutorBase):
    """Execute using parsl pyapp wrapper

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : Accumulatable
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

    tailtimeout: Optional[int] = None
    config: Optional["parsl.config.Config"] = None  # noqa

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        if len(items) == 0:
            return accumulator
        import parsl
        from parsl.app.app import python_app
        from .parsl.timeout import timeout

        if self.compression is not None:
            function = _compression_wrapper(self.compression, function)

        cleanup = False
        try:
            parsl.dfk()
        except RuntimeError:
            cleanup = True
            pass
        if cleanup and self.config is None:
            raise RuntimeError(
                "No active parsl DataFlowKernel, must specify a config to construct one"
            )
        elif not cleanup and self.config is not None:
            raise RuntimeError("An active parsl DataFlowKernel already exists")
        elif self.config is not None:
            parsl.clear()
            parsl.load(self.config)

        app = timeout(python_app(function))

        gen = _futures_handler(map(app, items), self.tailtimeout)
        try:
            accumulator = accumulate(
                tqdm(
                    gen if self.compression is None else map(_decompress, gen),
                    disable=not self.status,
                    unit=self.unit,
                    total=len(items),
                    desc=self.desc,
                ),
                accumulator,
            )
        finally:
            gen.close()

        if cleanup:
            parsl.dfk().cleanup()
            parsl.clear()

        return accumulator


class ParquetFileContext:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


@dataclass
class Runner:
    """A tool to run a processor using uproot for data delivery

    A convenience wrapper to submit jobs for a file set, which is a
    dictionary of dataset: [file list] entries.  Supports only uproot TTree
    reading, via NanoEvents or LazyDataFrame.  For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.

    Parameters
    ----------
        executor : ExecutorBase instance
            Executor, which implements a callable with inputs: items, function, accumulator
            and performs some action equivalent to:
            ``for item in items: accumulator += function(item)``
        pre_executor : ExecutorBase instance
            Executor, used to calculate fileset metadata
            Defaults to executor
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
        dynamic_chunksize : dict, optional
            Whether to adapt the chunksize for units of work to run in the targets given.
            Currently supported are 'wall_time' (in seconds), and 'memory' (in MB).
            E.g., with {"wall_time": 120, "memory": 2048}, the chunksize will
            be dynamically adapted so that processing jobs each run in about
            two minutes, using two GB of memory. (Currently only for the WorkQueueExecutor.)
    """

    executor: ExecutorBase
    pre_executor: Optional[ExecutorBase] = None
    chunksize: int = 100000
    maxchunks: Optional[int] = None
    metadata_cache: Optional[MutableMapping] = None
    dynamic_chunksize: Optional[Dict] = None
    skipbadfiles: bool = False
    xrootdtimeout: Optional[int] = None
    align_clusters: bool = False
    savemetrics: bool = False
    mmap: bool = False
    schema: Optional[schemas.BaseSchema] = schemas.BaseSchema
    cachestrategy: Optional[Union[Literal["dask-worker"], Callable[..., MutableMapping]]] = None  # fmt: skip
    processor_compression: int = 1
    use_skyhook: Optional[bool] = False
    skyhook_options: Optional[Dict] = field(default_factory=dict)
    format: str = "root"

    def __post_init__(self):
        if self.pre_executor is None:
            self.pre_executor = self.executor

        assert isinstance(
            self.executor, ExecutorBase
        ), "Expected executor to derive from ExecutorBase"
        assert isinstance(
            self.pre_executor, ExecutorBase
        ), "Expected pre_executor to derive from ExecutorBase"

        if self.metadata_cache is None:
            self.metadata_cache = DEFAULT_METADATA_CACHE

        if self.align_clusters and self.dynamic_chunksize:
            raise RuntimeError(
                "align_clusters and dynamic_chunksize cannot be used simultaneously"
            )
        if self.maxchunks and self.dynamic_chunksize:
            raise RuntimeError(
                "maxchunks and dynamic_chunksize cannot be used simultaneously"
            )
        if self.dynamic_chunksize and not isinstance(self.executor, WorkQueueExecutor):
            raise RuntimeError(
                "dynamic_chunksize currently only supported by the WorkQueueExecutor"
            )

        assert self.format in ("root", "parquet")

    @property
    def retries(self):
        if isinstance(self.executor, DaskExecutor):
            retries = 0
        else:
            retries = getattr(self.executor, "retries", 0)
        assert retries >= 0
        return retries

    @property
    def use_dataframes(self):
        if isinstance(self.executor, DaskExecutor):
            return self.executor.use_dataframes
        else:
            return False

    @staticmethod
    def get_cache(cachestrategy):
        cache = None
        if cachestrategy == "dask-worker":
            from distributed import get_worker
            from coffea.processor.dask import ColumnCache

            worker = get_worker()
            try:
                cache = worker.plugins[ColumnCache.name]
            except KeyError:
                # emit warning if not found?
                pass
        elif callable(cachestrategy):
            cache = cachestrategy()
        return cache

    @staticmethod
    def automatic_retries(retries: int, skipbadfiles: bool, func, *args, **kwargs):
        """This should probably defined on Executor-level."""
        import warnings

        retry_count = 0
        while retry_count <= retries:
            try:
                return func(*args, **kwargs)
            # catch xrootd errors and optionally skip
            # or retry to read the file
            except Exception as e:
                if skipbadfiles and isinstance(e, FileNotFoundError):
                    warnings.warn(str(e))
                    break
                if (
                    not skipbadfiles
                    or "Auth failed" in str(e)
                    or retries == retry_count
                ):
                    raise e
                warnings.warn("Attempt %d of %d." % (retry_count + 1, retries + 1))
            retry_count += 1

    @staticmethod
    def _normalize_fileset(
        fileset: Dict,
        treename: str,
    ) -> Generator[FileMeta, None, None]:
        if isinstance(fileset, str):
            with open(fileset) as fin:
                fileset = json.load(fin)
        elif not isinstance(fileset, Mapping):
            raise ValueError("Expected fileset to be a path string or mapping")
        reserved_metakeys = _PROTECTED_NAMES
        for dataset, filelist in fileset.items():
            user_meta = None
            if isinstance(filelist, dict):
                user_meta = filelist["metadata"] if "metadata" in filelist else None
                if user_meta is not None:
                    for rkey in reserved_metakeys:
                        if rkey in user_meta.keys():
                            raise ValueError(
                                f'Reserved word "{rkey}" in metadata section of fileset dictionary, please rename this entry!'
                            )
                if "treename" not in filelist and treename is None:
                    raise ValueError(
                        "treename must be specified if the fileset does not contain tree names"
                    )
                local_treename = (
                    filelist["treename"] if "treename" in filelist else treename
                )
                filelist = filelist["files"]
            elif isinstance(filelist, list):
                if treename is None:
                    raise ValueError(
                        "treename must be specified if the fileset does not contain tree names"
                    )
                local_treename = treename
            else:
                raise ValueError(
                    "list of filenames in fileset must be a list or a dict"
                )
            for filename in filelist:
                yield FileMeta(dataset, filename, local_treename, user_meta)

    @staticmethod
    def metadata_fetcher(
        xrootdtimeout: int, align_clusters: bool, item: FileMeta
    ) -> Accumulatable:
        out = set_accumulator()
        file = uproot.open(item.filename, timeout=xrootdtimeout)
        tree = file[item.treename]
        metadata = {}
        if item.metadata:
            metadata.update(item.metadata)
        metadata.update({"numentries": tree.num_entries, "uuid": file.file.fUUID})
        if align_clusters:
            metadata["clusters"] = tree.common_entry_offsets()
        out = set_accumulator(
            [FileMeta(item.dataset, item.filename, item.treename, metadata)]
        )
        return out

    def _preprocess_fileset(self, fileset: Dict) -> None:
        # this is a bit of an abuse of map-reduce but ok
        to_get = set(
            filemeta
            for filemeta in fileset
            if not filemeta.populated(clusters=self.align_clusters)
        )
        if len(to_get) > 0:
            out = set_accumulator()
            pre_arg_override = {
                "function_name": "get_metadata",
                "desc": "Preprocessing",
                "unit": "file",
                "compression": None,
            }
            if isinstance(self.pre_executor, (FuturesExecutor, ParslExecutor)):
                pre_arg_override.update({"tailtimeout": None})
            if isinstance(self.pre_executor, (DaskExecutor)):
                self.pre_executor.heavy_input = None
                pre_arg_override.update({"worker_affinity": False})
            pre_executor = self.pre_executor.copy(**pre_arg_override)
            closure = partial(
                self.automatic_retries,
                self.retries,
                self.skipbadfiles,
                partial(self.metadata_fetcher, self.xrootdtimeout, self.align_clusters),
            )
            out = pre_executor(to_get, closure, out)
            while out:
                item = out.pop()
                self.metadata_cache[item] = item.metadata
            for filemeta in fileset:
                filemeta.maybe_populate(self.metadata_cache)

    def _filter_badfiles(self, fileset: Dict) -> List:
        final_fileset = []
        for filemeta in fileset:
            if filemeta.populated(clusters=self.align_clusters):
                final_fileset.append(filemeta)
            elif not self.skipbadfiles:
                raise RuntimeError("Metadata for file {} could not be accessed.")
        return final_fileset

    def _chunk_generator(self, fileset: Dict, treename: str) -> Generator:
        if self.format == "root":
            if self.maxchunks is None:
                last_chunksize = self.chunksize
                for filemeta in fileset:
                    last_chunksize = yield from filemeta.chunks(
                        last_chunksize,
                        self.align_clusters,
                        self.dynamic_chunksize,
                    )
            else:
                # get just enough file info to compute chunking
                nchunks = defaultdict(int)
                chunks = []
                for filemeta in fileset:
                    if nchunks[filemeta.dataset] >= self.maxchunks:
                        continue
                    for chunk in filemeta.chunks(
                        self.chunksize, self.align_clusters, dynamic_chunksize=None
                    ):
                        chunks.append(chunk)
                        nchunks[filemeta.dataset] += 1
                        if nchunks[filemeta.dataset] >= self.maxchunks:
                            break
                yield from iter(chunks)
        else:
            import pyarrow.dataset as ds

            dataset_filelist_map = {}
            for dataset, basedir in fileset.items():
                ds_ = ds.dataset(basedir, format="parquet")
                dataset_filelist_map[dataset] = ds_.files
            chunks = []
            for dataset, filelist in dataset_filelist_map.items():
                for filename in filelist:
                    # If skyhook config is provided and is not empty,
                    if self.use_skyhook:
                        ceph_config_path = self.skyhook_options.get(
                            "ceph_config_path", "/etc/ceph/ceph.conf"
                        )
                        ceph_data_pool = self.skyhook_options.get(
                            "ceph_data_pool", "cephfs_data"
                        )
                        filename = f"{ceph_config_path}:{ceph_data_pool}:{filename}"
                    chunks.append(WorkItem(dataset, filename, treename, 0, 0, ""))
            yield from iter(chunks)

    @staticmethod
    def _work_function(
        format: str,
        xrootdtimeout: int,
        mmap: bool,
        schema: schemas.BaseSchema,
        cache_function: Callable[[], MutableMapping],
        use_dataframes: bool,
        savemetrics: bool,
        item: WorkItem,
        processor_instance: ProcessorABC,
    ) -> Dict:
        if processor_instance == "heavy":
            item, processor_instance = item
        if not isinstance(processor_instance, ProcessorABC):
            processor_instance = cloudpickle.loads(lz4f.decompress(processor_instance))

        if format == "root":
            filecontext = uproot.open(
                item.filename,
                timeout=xrootdtimeout,
                file_handler=uproot.MemmapSource
                if mmap
                else uproot.MultithreadedFileSource,
            )
        elif format == "parquet":
            filecontext = ParquetFileContext(item.filename)

        metadata = {
            "dataset": item.dataset,
            "filename": item.filename,
            "treename": item.treename,
            "entrystart": item.entrystart,
            "entrystop": item.entrystop,
            "fileuuid": str(uuid.UUID(bytes=item.fileuuid))
            if len(item.fileuuid) > 0
            else "",
        }
        if item.usermeta is not None:
            metadata.update(item.usermeta)

        with filecontext as file:
            if schema is None:
                # To deprecate
                tree = file[item.treename]
                events = LazyDataFrame(
                    tree, item.entrystart, item.entrystop, metadata=metadata
                )
            elif issubclass(schema, schemas.BaseSchema):
                # change here
                if format == "root":
                    materialized = []
                    factory = NanoEventsFactory.from_root(
                        file=file,
                        treepath=item.treename,
                        entry_start=item.entrystart,
                        entry_stop=item.entrystop,
                        persistent_cache=cache_function(),
                        schemaclass=schema,
                        metadata=metadata,
                        access_log=materialized,
                    )
                    events = factory.events()
                elif format == "parquet":
                    skyhook_options = {}
                    if ":" in item.filename:
                        (
                            ceph_config_path,
                            ceph_data_pool,
                            filename,
                        ) = item.filename.split(":")
                        # patch back filename into item
                        item = WorkItem(**dict(asdict(item), filename=filename))
                        skyhook_options["ceph_config_path"] = ceph_config_path
                        skyhook_options["ceph_data_pool"] = ceph_data_pool

                    factory = NanoEventsFactory.from_parquet(
                        file=item.filename,
                        treepath=item.treename,
                        schemaclass=schema,
                        metadata=metadata,
                        skyhook_options=skyhook_options,
                    )
                    events = factory.events()
            else:
                raise ValueError(
                    "Expected schema to derive from nanoevents.BaseSchema, instead got %r"
                    % schema
                )
            tic = time.time()
            try:
                out = processor_instance.process(events)
            except Exception as e:
                file_trace = f"\n\nFailed processing file: {item!r}"
                raise type(e)(str(e) + file_trace).with_traceback(
                    sys.exc_info()[2]
                ) from None
            if out is None:
                raise ValueError(
                    "Output of process() should not be None. Make sure your processor's process() function returns an accumulator."
                )
            toc = time.time()
            if use_dataframes:
                return out
            else:
                if savemetrics:
                    metrics = {}
                    if isinstance(file, uproot.ReadOnlyDirectory):
                        metrics["bytesread"] = file.file.source.num_requested_bytes
                    if schema is not None and issubclass(schema, schemas.BaseSchema):
                        metrics["columns"] = set(materialized)
                        metrics["entries"] = len(events)
                    else:
                        metrics["columns"] = set(events.materialized)
                        metrics["entries"] = events.size
                    metrics["processtime"] = toc - tic
                    return {"out": out, "metrics": metrics}
                return {"out": out}

    def __call__(
        self,
        fileset: Dict,
        treename: str,
        processor_instance: ProcessorABC,
    ) -> Accumulatable:
        """Run the processor_instance on a given fileset

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
        """

        if not isinstance(fileset, (Mapping, str)):
            raise ValueError(
                "Expected fileset to be a mapping dataset: list(files) or filename"
            )
        if not isinstance(processor_instance, ProcessorABC):
            raise ValueError("Expected processor_instance to derive from ProcessorABC")

        if self.format == "root":
            fileset = list(self._normalize_fileset(fileset, treename))
            for filemeta in fileset:
                filemeta.maybe_populate(self.metadata_cache)

            self._preprocess_fileset(fileset)
            fileset = self._filter_badfiles(fileset)

            # reverse fileset list to match the order of files as presented in version
            # v0.7.4. This fixes tests using maxchunks.
            fileset.reverse()

        chunks = self._chunk_generator(fileset, treename)

        if self.processor_compression is None:
            pi_to_send = processor_instance
        else:
            pi_to_send = lz4f.compress(
                cloudpickle.dumps(processor_instance),
                compression_level=self.processor_compression,
            )
        # hack around dask/dask#5503 which is really a silly request but here we are
        if isinstance(self.executor, DaskExecutor):
            self.executor.heavy_input = pi_to_send
            closure = partial(
                self._work_function,
                self.format,
                self.xrootdtimeout,
                self.mmap,
                self.schema,
                partial(self.get_cache, self.cachestrategy),
                self.use_dataframes,
                self.savemetrics,
                processor_instance="heavy",
            )
        else:
            closure = partial(
                self._work_function,
                self.format,
                self.xrootdtimeout,
                self.mmap,
                self.schema,
                partial(self.get_cache, self.cachestrategy),
                self.use_dataframes,
                self.savemetrics,
                processor_instance=pi_to_send,
            )

        if self.format == "root":
            if self.dynamic_chunksize:
                events_total = sum(f.metadata["numentries"] for f in fileset)
            else:
                chunks = [c for c in chunks]
                events_total = sum(len(c) for c in chunks)
        else:
            chunks = [c for c in chunks]

        exe_args = {
            "unit": "event" if isinstance(self.executor, WorkQueueExecutor) else "chunk",  # fmt: skip
            "function_name": type(processor_instance).__name__,
        }
        if self.format == "root" and isinstance(self.executor, WorkQueueExecutor):
            exe_args.update(
                {
                    "events_total": events_total,
                    "dynamic_chunksize": self.dynamic_chunksize,
                    "chunksize": self.chunksize,
                }
            )

        closure = partial(
            self.automatic_retries, self.retries, self.skipbadfiles, closure
        )

        executor = self.executor.copy(**exe_args)
        wrapped_out = executor(chunks, closure, None)

        processor_instance.postprocess(wrapped_out["out"])
        if self.savemetrics and not self.use_dataframes:
            wrapped_out["metrics"]["chunks"] = len(chunks)
            return wrapped_out["out"], wrapped_out["metrics"]
        return wrapped_out["out"]


def run_spark_job(
    fileset,
    processor_instance,
    executor,
    executor_args={},
    spark=None,
    partitionsize=200000,
    thread_workers=16,
):
    """A wrapper to submit spark jobs

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
    """

    try:
        import pyspark
    except ImportError as e:
        print(
            "you must have pyspark installed to call run_spark_job()!", file=sys.stderr
        )
        raise e

    from packaging import version
    import pyarrow as pa
    import warnings

    arrow_env = ("ARROW_PRE_0_15_IPC_FORMAT", "1")
    if version.parse(pa.__version__) >= version.parse("0.15.0") and version.parse(
        pyspark.__version__
    ) < version.parse("3.0.0"):
        import os

        if arrow_env[0] not in os.environ or os.environ[arrow_env[0]] != arrow_env[1]:
            warnings.warn(
                "If you are using pyarrow >= 0.15.0, make sure to set %s=%s in your environment!"
                % arrow_env
            )

    import pyspark.sql
    from .spark.spark_executor import SparkExecutor
    from .spark.detail import _spark_initialize, _spark_stop, _spark_make_dfs

    if not isinstance(fileset, Mapping):
        raise ValueError("Expected fileset to be a mapping dataset: list(files)")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")
    if not isinstance(executor, SparkExecutor):
        raise ValueError("Expected executor to derive from SparkExecutor")

    executor_args.setdefault("config", None)
    executor_args.setdefault("file_type", "parquet")
    executor_args.setdefault("laurelin_version", "1.1.1")
    executor_args.setdefault("treeName", "Events")
    executor_args.setdefault("schema", None)
    executor_args.setdefault("cache", True)
    executor_args.setdefault("skipbadfiles", False)
    executor_args.setdefault("retries", 0)
    executor_args.setdefault("xrootdtimeout", None)
    file_type = executor_args["file_type"]
    treeName = executor_args["treeName"]
    schema = executor_args["schema"]
    if "flatten" in executor_args:
        raise ValueError(
            "Executor argument 'flatten' is deprecated, please refactor your processor to accept awkward arrays"
        )
    if "nano" in executor_args:
        raise ValueError(
            "Awkward0 NanoEvents no longer supported.\n"
            "Please use 'schema': processor.NanoAODSchema to enable awkward NanoEvents processing."
        )
    use_cache = executor_args["cache"]

    if executor_args["config"] is None:
        executor_args.pop("config")

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
            raise ValueError(
                "Expected 'spark' to be a pyspark.sql.session.SparkSession"
            )

    dfslist = {}
    if executor._cacheddfs is None:
        dfslist = _spark_make_dfs(
            spark,
            fileset,
            partitionsize,
            processor_instance.columns,
            thread_workers,
            file_type,
            treeName,
        )

    output = executor(
        spark, dfslist, processor_instance, None, thread_workers, use_cache, schema
    )
    processor_instance.postprocess(output)

    if killSpark:
        _spark_stop(spark)
        del spark
        spark = None

    return output
