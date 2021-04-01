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
import uproot
import subprocess
import re
import os
import uuid
import warnings
from tqdm.auto import tqdm
from collections import defaultdict
from cachetools import LRUCache
import lz4.frame as lz4f
from .processor import ProcessorABC
from .accumulator import (
    accumulate,
    Accumulatable,
    set_accumulator,
)
from .dataframe import (
    LazyDataFrame,
)
from ..nanoevents import NanoEventsFactory, schemas
from ..util import _hash

from collections.abc import Mapping, MutableMapping, Sequence


_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
DEFAULT_METADATA_CACHE: MutableMapping = LRUCache(100000)


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
    __slots__ = [
        'dataset',
        'filename',
        'treename',
        'entrystart',
        'entrystop',
        'fileuuid',
    ]

    def __init__(self, dataset, filename, treename, entrystart, entrystop, fileuuid):
        self.dataset = dataset
        self.filename = filename
        self.treename = treename
        self.entrystart = entrystart
        self.entrystop = entrystop
        self.fileuuid = fileuuid


def _compress(item, clevel):
    return lz4f.compress(pickle.dumps(item, protocol=_PICKLE_PROTOCOL), compression_level=clevel)


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
    def __init__(self, clevel):
        self.clevel = clevel

    def __str__(self):
        return "reduce"

    def __call__(self, items):
        items = list(items)
        if len(items) == 0:
            raise ValueError("Empty list provided to reduction")
        if self.clevel is not None:
            out = _decompress(items.pop())
            out = accumulate(map(_decompress, items), out)
            return _compress(out, self.clevel)
        return accumulate(items)


def _cancel(job):
    try:
        # this is not implemented with parsl AppFutures
        job.cancel()
    except NotImplementedError:
        pass


def _futures_handler(futures, timeout):
    """Essentially the same as concurrent.futures.as_completed
    but makes sure not to hold references to futures any longer than strictly necessary,
    which is important if the future holds a large result.
    """
    futures = set(futures)
    try:
        while futures:
            try:
                done, futures = concurrent.futures.wait(futures, timeout=timeout, return_when=concurrent.futures.FIRST_COMPLETED)
                if len(done) == 0:
                    warnings.warn(f"No finished jobs after {timeout}s, stopping remaining {len(futures)} jobs early")
                    break
                while done:
                    try:
                        yield done.pop().result()
                    except concurrent.futures.CancelledError:
                        pass
            except KeyboardInterrupt:
                for job in futures:
                    _cancel(job)
                running = sum(job.running() for job in futures)
                warnings.warn(f"Early stop: cancelled {len(futures) - running} jobs, will wait for {running} running jobs to complete")
    finally:
        running = sum(job.running() for job in futures)
        if running:
            warnings.warn(f"Cancelling {running} running jobs (likely due to an exception)")
        while futures:
            _cancel(futures.pop())


def wqex_create_function_wrapper(tmpdir):
    """ Writes a wrapper script to run dilled python functions and arguments.
    The wrapper takes as arguments the name of three files: function, argument, and output.
    The files function and argument have the dilled function and argument, respectively.
    The file output is created (or overwritten), with the dilled result of the function call.
    The wrapper created is created/deleted according to the lifetime of the work_queue_executor."""

    name = os.path.join(tmpdir, 'fn_as_file')

    with open(name, mode='w') as f:
        f.write("""
#!/usr/bin/env python3
import os
import sys
import dill
import coffea

(fn, arg, out) = sys.argv[1], sys.argv[2], sys.argv[3]

with open(fn, "rb") as f:
    exec_function = dill.load(f)
with open(arg, "rb") as f:
    exec_item = dill.load(f)

pickle_out = exec_function(exec_item)
with open(out, "wb") as f:
    dill.dump(pickle_out, f)

# Force an OS exit here to avoid a bug in xrootd finalization
os._exit(0)
""")

    return name


def wqex_create_task(itemid, item, wrapper, env_file, command_path, infile_function, tmpdir, extra_input_files):
    import dill
    from os.path import basename
    import work_queue as wq

    with open(os.path.join(tmpdir, 'item_{}.p'.format(itemid)), 'wb') as wf:
        dill.dump(item, wf)

    infile_item = os.path.join(tmpdir, 'item_{}.p'.format(itemid))
    outfile = os.path.join(tmpdir, 'output_{}.p'.format(itemid))

    # Base command just invokes python on the function and data.
    command = 'python {} {} {} {}'.format(basename(command_path), basename(infile_function), basename(infile_item), basename(outfile))

    # If wrapper and env provided, add that.
    if wrapper and env_file:
        command = './{} --environment {} --unpack-to "$WORK_QUEUE_SANDBOX"/{}-env {}'.format(basename(wrapper), basename(env_file), basename(env_file), command)

    task = wq.Task(command)
    task.specify_category('default')
    task.specify_input_file(command_path, cache=True)
    task.specify_input_file(infile_function, cache=False)
    task.specify_input_file(infile_item, cache=False)

    for f in extra_input_files:
        task.specify_input_file(f, cache=True)

    if wrapper and env_file:
        task.specify_input_file(env_file, cache=True)
        task.specify_input_file(wrapper, cache=True)

    if re.search('://', item.filename):
        # This looks like an URL. Not transfering file.
        pass
    else:
        task.specify_input_file(item.filename, remote_name=item.filename, cache=True)

    task.specify_output_file(outfile, cache=False)

    # Put the item ID into the tag to associate upon completion.
    task.specify_tag("{}".format(itemid))

    return task


def wqex_output_task(task, verbose_mode, resource_mode, output_mode):
    if verbose_mode:
        print('Task (id #{}) complete: {} (return code {})'.format(task.id, task.command, task.return_status))

        print('Allocated cores: {}, memory: {} MB, disk: {} MB, gpus: {}'.format(
            task.resources_allocated.cores,
            task.resources_allocated.memory,
            task.resources_allocated.disk,
            task.resources_allocated.gpus))

        if resource_mode:
            print('Measured cores: {}, memory: {} MB, disk {} MB, gpus: {}, runtime {}'.format(
                task.resources_measured.cores,
                task.resources_measured.memory,
                task.resources_measured.disk,
                task.resources_measures.gpus,
                task.resources_measured.wall_time / 1000000))

    if output_mode and task.output:
        print('Task id #{} output:\n{}'.format(task.id, task.output))

    if task.result != 0:
        print('Task id #{} failed with code: {}'.format(task.id, task.result))


# The Work Queue object is global b/c we want to
# retain state between runs of the executor, such
# as connections to workers, cached data, etc.
_wq_queue = None


def work_queue_executor(items, function, accumulator, **kwargs):
    """Execute using Work Queue

    For more information, see :ref:`intro-coffea-wq`

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
            Compress accumulator outputs in flight with LZ4, at level specified (default 1)
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

        resources-mode : one of 'fixed', or 'auto'. Default is 'fixed'.
            - 'fixed': allocate cores, memory, and disk specified for each task.
            - 'auto': use cores, memory, and disk as maximum values to allocate.
                      Useful when the resources used by a task are not known, as
                      it lets work queue find an efficient value for maximum
                      throughput.
        resource-monitor : bool
            If true, (false is the default) turns on resource monitoring for Work Queue.

        master-name : str
            Name to refer to this work queue master.
            Sets port to 0 (any available port) if port not given.
        port : int
            Port number for work queue master program. Defaults to 9123 if
            master-name not given.
        password-file: str
            Location of a file containing a password used to authenticate workers.

        extra-input-files: list
            A list of files in the current working directory to send along with each task.
            Useful for small custom libraries and configuration files needed by the processor.
        environment-file : str
            Python environment to use. Required.
        wrapper : str
            Wrapper script to run/open python environment tarball. Defaults to python_package_run found in PATH.

        verbose : bool
            If true, emit a message on each task submission and completion.
            Default is false.
        debug-log : str
            Filename for debug output
        stats-log : str
            Filename for tasks statistics output
        transactions-log : str
            Filename for tasks lifetime reports output
        print-stdout : bool
            If true (default), print the standard output of work queue task on completion.
    """
    try:
        import work_queue as wq
        import tempfile
        import dill
        import os
        from os.path import basename
    except ImportError as e:
        print('You must have Work Queue and dill installed to use work_queue_executor!')
        raise e

    global _wq_queue

    # Standard executor options:
    unit = kwargs.pop('unit', 'items')
    status = kwargs.pop('status', True)
    desc = kwargs.pop('desc', 'Processing')
    clevel = kwargs.pop('compression', 1)
    filepath = kwargs.pop('filepath', '.')

    if clevel is not None:
        function = _compression_wrapper(clevel, function)

    # Work Queue specific options:
    verbose_mode = kwargs.pop('verbose', False)
    debug_log = kwargs.pop('debug-log', None)
    stats_log = kwargs.pop('stats-log', None)
    trans_log = kwargs.pop('transactions-log', None)
    extra_input_files = kwargs.pop('extra-input-files', [])
    output = kwargs.pop('print-stdout', False)
    password_file = kwargs.pop('password-file', None)
    env_file = kwargs.pop('environment-file', None)
    wrapper = kwargs.pop('wrapper', shutil.which('python_package_run'))
    resources_mode = kwargs.pop('resources-mode', 'fixed')
    cores = kwargs.pop('cores', None)
    memory = kwargs.pop('memory', None)
    disk = kwargs.pop('disk', None)
    gpus = kwargs.pop('gpus', None)
    resource_monitor = kwargs.pop('resource-monitor', False)
    master_name = kwargs.pop('master-name', None)
    port = kwargs.pop('port', None)
    if port is None:
        if master_name:
            port = 0
        else:
            port = 9123

    if _wq_queue is None:
        _wq_queue = wq.WorkQueue(port, name=master_name, debug_log=debug_log, stats_log=stats_log, transactions_log=trans_log)

    if env_file and not wrapper:
        raise ValueError("Location of python_package_run could not be determined automatically.\nUse 'wrapper' argument to the work_queue_executor.")

    # If explicit resources are given, collect them into default_resources
    default_resources = {}
    if cores:
        default_resources['cores'] = cores
    if memory:
        default_resources['memory'] = memory
    if disk:
        default_resources['disk'] = disk
    if gpus:
        default_resources['gpus'] = gpus

    # Working within a custom temporary directory:
    with tempfile.TemporaryDirectory(prefix="wq-executor-tmp-", dir=filepath) as tmpdir:

        # Save the executable function in a dilled file.
        with open(os.path.join(tmpdir, 'function.p'), 'wb') as wf:
            dill.dump(function, wf)

        # Create a wrapper script to run the function.
        command_path = wqex_create_function_wrapper(tmpdir)

        # Enable monitoring and auto resource consumption, if desired:
        if resource_monitor:
            _wq_queue.enable_monitoring()

        _wq_queue.specify_category_max_resources('default', default_resources)
        if resources_mode == 'auto':
            _wq_queue.tune('category-steady-n-tasks', 3)
            _wq_queue.specify_category_max_resources('default', {})
            _wq_queue.specify_category_mode('default', wq.WORK_QUEUE_ALLOCATION_MODE_MAX_THROUGHPUT)

        # Make use of the stored password file, if enabled.
        if password_file is not None:
            _wq_queue.specify_password_file(password_file)

        infile_function = os.path.join(tmpdir, 'function.p')

        # Nothing to do?  Just return the accumulator.
        if len(items) == 0:
            return accumulator

        # Keep track of total tasks in each state.
        tasks_submitted = 0
        tasks_done = 0
        tasks_total = len(items)

        print('Listening for work queue workers on port {}...'.format(_wq_queue.port))

        # Create a dual progress bar to show submission and completion.
        submit_bar = tqdm(total=tasks_total, position=0, disable=not status, unit=unit, desc="Submitted")
        complete_bar = tqdm(total=tasks_total, position=1, disable=not status, unit=unit, desc=desc)

        itemiter = iter(items)

        # Main loop of executor
        while tasks_done < tasks_total:

            # Submit tasks into the queue, but no more than 100 idle tasks
            while tasks_submitted < tasks_total and _wq_queue.stats.tasks_waiting < 100:
                item = next(itemiter)
                task = wqex_create_task(tasks_submitted, item, wrapper, env_file, command_path, infile_function, tmpdir, extra_input_files)
                task_id = _wq_queue.submit(task)
                tasks_submitted += 1

                if(verbose_mode):
                    print('Submitted task (id #{}): {}'.format(task_id, task.command))
                else:
                    submit_bar.update(1)
                    complete_bar.update(0)

            # When done submitting, look for completed tasks.

            task = _wq_queue.wait(5)
            if task:
                # Display details of the completed task
                wqex_output_task(task, verbose_mode, resource_monitor, output)
                if task.result != 0:
                    # Note that WQ already retries internal failures.
                    # If we get to this point, it's a badly formed task.
                    raise RuntimeError("Task {} item {} failed with output:\n{}".format(task.id, task.tag, task.output))

                # The task tag remembers the itemid for us.
                itemid = task.tag
                itemfile = os.path.join(tmpdir, 'item_{}.p'.format(itemid))
                outfile = os.path.join(tmpdir, 'output_{}.p'.format(itemid))

                # Accumulate results from the pickled output
                with open(outfile, 'rb') as rf:
                    unpickle_output = dill.load(rf)
                accumulator = accumulate([unpickle_output if clevel is None else _decompress(unpickle_output)], accumulator)

                # Remove output files as we go to avoid unbounded disk
                os.remove(itemfile)
                os.remove(outfile)

                tasks_done += 1

                if(not verbose_mode):
                    submit_bar.update(0)
                    complete_bar.update(1)

        submit_bar.close()
        complete_bar.close()

        return accumulator


def iterative_executor(items, function, accumulator, **kwargs):
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
    if len(items) == 0:
        return accumulator
    status = kwargs.pop('status', True)
    unit = kwargs.pop('unit', 'items')
    desc = kwargs.pop('desc', 'Processing')
    gen = tqdm(items, disable=not status, unit=unit, total=len(items), desc=desc)
    gen = map(function, gen)
    return accumulate(gen, accumulator)


def futures_executor(items, function, accumulator, **kwargs):
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

    def processwith(pool):
        gen = _futures_handler({pool.submit(function, item) for item in items}, tailtimeout)
        try:
            return accumulate(
                tqdm(
                    gen if clevel is None else map(_decompress, gen),
                    disable=not status,
                    unit=unit,
                    total=len(items),
                    desc=desc,
                ),
                accumulator
            )
        finally:
            gen.close()

    if isinstance(pool, concurrent.futures.Executor):
        return processwith(pool)
    else:
        # assume its a class then
        with pool(max_workers=workers) as poolinstance:
            return processwith(poolinstance)


def dask_executor(items, function, accumulator, **kwargs):
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
    import pandas as pd
    import dask.dataframe as dd
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
    use_dataframes = kwargs.pop('use_dataframes', False)
    # secret options
    worker_affinity = kwargs.pop('worker_affinity', False)

    if use_dataframes:
        clevel = None

    reducer = _reduce(clevel)
    if clevel is not None:
        function = _compression_wrapper(clevel, function, name=function_name)

    if heavy_input is not None:
        # client.scatter is not robust against adaptive clusters
        # https://github.com/CoffeaTeam/coffea/issues/465
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Large object of size")
            items = list(zip(items, repeat(client.submit(lambda x: x, heavy_input))))

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
    if (function_name == 'get_metadata') or not use_dataframes:
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
        return accumulate([work.result() if clevel is None else _decompress(work.result())], accumulator)
    else:
        if status:
            from distributed import progress
            progress(work, multi=True, notebook=False)
        return {"out": dd.from_delayed(work)}


def parsl_executor(items, function, accumulator, **kwargs):
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

    gen = _futures_handler(map(app, items), tailtimeout)
    try:
        accumulator = accumulate(
            tqdm(
                gen if clevel is None else map(_decompress, gen),
                disable=not status,
                unit=unit,
                total=len(items),
                desc=desc,
            ),
            accumulator
        )
    finally:
        gen.close()

    if cleanup:
        parsl.dfk().cleanup()
        parsl.clear()

    return accumulator


def _get_cache(strategy):
    cache = None
    if strategy == 'dask-worker':
        from distributed import get_worker
        from coffea.processor.dask import ColumnCache
        worker = get_worker()
        try:
            cache = worker.plugins[ColumnCache.name]
        except KeyError:
            # emit warning if not found?
            pass
    elif callable(strategy):
        cache = strategy()

    return cache


class ParquetFileContext():
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass


def _work_function(item, processor_instance, savemetrics=False,
                   mmap=False, schema=schemas.BaseSchema, cachestrategy=None, skipbadfiles=False,
                   retries=0, xrootdtimeout=None, use_dataframes=False, format="root"):
    if processor_instance == 'heavy':
        item, processor_instance = item
    if not isinstance(processor_instance, ProcessorABC):
        processor_instance = cloudpickle.loads(lz4f.decompress(processor_instance))

    retry_count = 0
    while retry_count <= retries:
        try:
            if format == "root":
                filecontext = uproot.open(
                    item.filename,
                    timeout=xrootdtimeout,
                    file_handler=uproot.MemmapSource if mmap else uproot.MultithreadedFileSource,
                )
            elif format == "parquet":
                filecontext = ParquetFileContext(item.filename)

            metadata = {
                'dataset': item.dataset,
                'filename': item.filename,
                'treename': item.treename,
                'entrystart': item.entrystart,
                'entrystop': item.entrystop,
                'fileuuid': str(uuid.UUID(bytes=item.fileuuid)) if len(item.fileuuid) > 0 else ''
            }

            with filecontext as file:
                if schema is None:
                    # To deprecate
                    tree = file[item.treename]
                    events = LazyDataFrame(tree, item.entrystart, item.entrystop, metadata=metadata)
                elif issubclass(schema, schemas.BaseSchema):
                    # change here
                    if format == "root":
                        materialized = []
                        factory = NanoEventsFactory.from_root(
                            file=file,
                            treepath=item.treename,
                            entry_start=item.entrystart,
                            entry_stop=item.entrystop,
                            persistent_cache=_get_cache(cachestrategy),
                            schemaclass=schema,
                            metadata=metadata,
                            access_log=materialized,
                        )
                        events = factory.events()
                    elif format == "parquet":
                        rados_parquet_options = {}
                        if ':' in item.filename:
                            ceph_config_path, item.filename = item.filename.split(":")
                            rados_parquet_options['ceph_config_path'] = ceph_config_path

                        factory = NanoEventsFactory.from_parquet(
                            file=item.filename,
                            treepath=item.treename,
                            metadata=metadata,
                            rados_parquet_options=rados_parquet_options
                        )
                        events = factory.events()
                else:
                    raise ValueError("Expected schema to derive from nanoevents.BaseSchema, instead got %r" % schema)
                tic = time.time()
                try:
                    out = processor_instance.process(events)
                except Exception as e:
                    file_trace = f"\n\nFailed processing file: {item.filename} ({item.entrystart}-{item.entrystop})"
                    raise type(e)(str(e) + file_trace).with_traceback(sys.exc_info()[2]) from None
                if out is None:
                    raise ValueError("Output of process() should not be None. Make sure your processor's process() function returns an accumulator.")
                toc = time.time()
                if use_dataframes:
                    return out
                else:
                    if savemetrics:
                        metrics = {}
                        if isinstance(file, uproot.ReadOnlyDirectory):
                            metrics['bytesread'] = file.file.source.num_requested_bytes
                        if schema is not None and issubclass(schema, schemas.BaseSchema):
                            metrics['columns'] = set(materialized)
                            metrics['entries'] = len(events)
                        else:
                            metrics['columns'] = set(events.materialized)
                            metrics['entries'] = events.size
                        metrics['processtime'] = toc - tic
                        return {'out': out, 'metrics': metrics}
                    return {"out": out}
            break
        # catch xrootd errors and optionally skip
        # or retry to read the file
        except OSError as e:
            if not skipbadfiles or "Auth failed" in str(e):
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
            if not use_dataframes:
                return {"out": out}
        except Exception as e:
            if retries == retry_count:
                raise e
            w_str = 'Attempt %d of %d. Will retry.' % (retry_count + 1, retries + 1)
            warnings.warn(w_str)
        retry_count += 1


def _normalize_fileset(fileset, treename):
    if isinstance(fileset, str):
        with open(fileset) as fin:
            fileset = json.load(fin)
    elif not isinstance(fileset, Mapping):
        raise ValueError("Expected fileset to be a path string or mapping")
    for dataset, filelist in fileset.items():
        if isinstance(filelist, dict):
            if set(filelist.keys()) - {"treename", "files"}:
                raise ValueError(f"Extraneous arguments in fileset dictionary, expected treename, files but got {set(filelist.keys())}")
            if "treename" not in filelist and treename is None:
                raise ValueError('treename must be specified if the fileset does not contain tree names')
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
            file = uproot.open(item.filename, timeout=xrootdtimeout)
            tree = file[item.treename]
            metadata = {'numentries': tree.num_entries, 'uuid': file.file.fUUID}
            if align_clusters:
                metadata['clusters'] = tree.common_entry_offsets()
            out = set_accumulator([FileMeta(item.dataset, item.filename, item.treename, metadata)])
            break
        except OSError as e:
            if not skipbadfiles or "Auth failed" in str(e):
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
    dictionary of dataset: [file list] entries.  Supports only uproot TTree
    reading, via NanoEvents or LazyDataFrame.  For more customized processing,
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
            Some options are not passed to executors but rather and affect the behavior of the
            work function itself:

            - ``savemetrics`` saves some detailed metrics for xrootd processing (default False)
            - ``schema`` builds the dataframe as a `nanoevents` object
              (default ``BaseSchema``); schema options include `BaseSchema`, `NanoAODSchema`, and `TreeMakerSchema`.
              If schema is None a `LazyDataFrame` is returned rather than NanoEvents, use for unruly ROOT files.
            - ``processor_compression`` sets the compression level used to send processor instance to workers (default 1)
            - ``skipbadfiles`` instead of failing on a bad file, skip it (default False)
            - ``retries`` optionally retry processing of a chunk on failure (default 0)
            - ``xrootdtimeout`` timeout for xrootd read (seconds)
            - ``tailtimeout`` timeout requirement on job tails (seconds)
            - ``align_clusters`` aligns the chunks to natural boundaries in the ROOT files (default False)
            - ``use_dataframes`` retrieve output as a distributed Dask DataFrame (default False).
                Only works with `dask_executor`; the processor output must be a Pandas DataFrame.
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
    if executor is dask_executor:
        # this executor has a builtin retry mechanism
        retries = 0
    else:
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
                'function_name': 'get_metadata',
                'desc': 'Preprocessing',
                'unit': 'file',
                'compression': None,
                'tailtimeout': None,
                'worker_affinity': False,
            }
            pre_args.update(pre_arg_override)
            out = pre_executor(to_get, metadata_fetcher, out, **pre_args)
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
    savemetrics = executor_args.pop('savemetrics', False)
    if "flatten" in executor_args:
        raise ValueError("Executor argument 'flatten' is deprecated, please refactor your processor to accept awkward arrays")
    mmap = executor_args.pop('mmap', False)
    schema = executor_args.pop('schema', schemas.BaseSchema)
    use_dataframes = executor_args.pop('use_dataframes', False)
    if (executor is not dask_executor) and use_dataframes:
        warnings.warn("Only Dask executor supports DataFrame outputs! Resetting 'use_dataframes' argument to False.")
        use_dataframes = False
    if "nano" in executor_args:
        raise ValueError("Awkward0 NanoEvents no longer supported.\n"
                         "Please use 'schema': processor.NanoAODSchema to enable awkward NanoEvents processing.")
    cachestrategy = executor_args.pop('cachestrategy', None)
    pi_compression = executor_args.pop('processor_compression', 1)
    if pi_compression is None:
        pi_to_send = processor_instance
    else:
        pi_to_send = lz4f.compress(cloudpickle.dumps(processor_instance), compression_level=pi_compression)
    closure = partial(
        _work_function,
        savemetrics=savemetrics,
        mmap=mmap,
        schema=schema,
        cachestrategy=cachestrategy,
        skipbadfiles=skipbadfiles,
        retries=retries,
        xrootdtimeout=xrootdtimeout,
        use_dataframes=use_dataframes,
    )
    # hack around dask/dask#5503 which is really a silly request but here we are
    if executor is dask_executor:
        executor_args['heavy_input'] = pi_to_send
        closure = partial(closure, processor_instance='heavy')
    else:
        closure = partial(closure, processor_instance=pi_to_send)

    exe_args = {
        'unit': 'chunk',
        'function_name': type(processor_instance).__name__,
        'use_dataframes': use_dataframes
    }
    exe_args.update(executor_args)
    wrapped_out = executor(chunks, closure, None, **exe_args)

    processor_instance.postprocess(wrapped_out["out"])
    if savemetrics and not use_dataframes:
        wrapped_out['metrics']['chunks'] = len(chunks)
        return wrapped_out["out"], wrapped_out['metrics']
    return wrapped_out["out"]


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
    executor_args.setdefault('laurelin_version', '1.1.1')
    executor_args.setdefault('treeName', 'Events')
    executor_args.setdefault('schema', None)
    executor_args.setdefault('cache', True)
    executor_args.setdefault('skipbadfiles', False)
    executor_args.setdefault('retries', 0)
    executor_args.setdefault('xrootdtimeout', None)
    file_type = executor_args['file_type']
    treeName = executor_args['treeName']
    schema = executor_args['schema']
    if "flatten" in executor_args:
        raise ValueError("Executor argument 'flatten' is deprecated, please refactor your processor to accept awkward arrays")
    if "nano" in executor_args:
        raise ValueError("Awkward0 NanoEvents no longer supported.\n"
                         "Please use 'schema': processor.NanoAODSchema to enable awkward NanoEvents processing.")
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

    output = executor(spark, dfslist, processor_instance, None, thread_workers, use_cache, schema)
    processor_instance.postprocess(output)

    if killSpark:
        _spark_stop(spark)
        del spark
        spark = None

    return output


def run_parquet_job(fileset, treename, processor_instance, executor, executor_args={}):
    import warnings
    import pyarrow.dataset as ds

    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")

    dataset_filelist_map = {}
    for dataset, basedir in fileset.items():
        ds_ = ds.dataset(basedir, format="parquet")
        dataset_filelist_map[dataset] = ds_.files

    # make a copy since we modify in-place
    executor_args = dict(executor_args)

    # pop _get_metdata args here (also sent to _work_function)
    skipbadfiles = executor_args.pop('skipbadfiles', False)
    if executor is dask_executor:
        # this executor has a builtin retry mechanism
        retries = 0
    else:
        retries = executor_args.pop('retries', 0)
    xrootdtimeout = executor_args.pop('xrootdtimeout', None)
    ceph_config_path = executor_args.pop('ceph_config_path', None)

    chunks = []
    for dataset, filelist in dataset_filelist_map.items():
        for filename in filelist:
            # if in cephfs, encode the ceph config path in the filename
            if ceph_config_path:
                filename = f"{ceph_config_path}:{filename}"
            chunks.append(WorkItem(dataset, filename, treename, 0, 0, ''))

    # pop all _work_function args here
    savemetrics = executor_args.pop('savemetrics', False)
    if "flatten" in executor_args:
        raise ValueError("Executor argument 'flatten' is deprecated, please refactor your processor to accept awkward arrays")
    mmap = executor_args.pop('mmap', False)
    schema = executor_args.pop('schema', schemas.BaseSchema)
    use_dataframes = executor_args.pop('use_dataframes', False)
    if (executor is not dask_executor) and use_dataframes:
        warnings.warn("Only Dask executor supports DataFrame outputs! Resetting 'use_dataframes' argument to False.")
        use_dataframes = False
    if "nano" in executor_args:
        raise ValueError("Awkward0 NanoEvents no longer supported.\n"
                         "Please use 'schema': processor.NanoAODSchema to enable awkward NanoEvents processing.")
    cachestrategy = executor_args.pop('cachestrategy', None)
    pi_compression = executor_args.pop('processor_compression', 1)
    if pi_compression is None:
        pi_to_send = processor_instance
    else:
        pi_to_send = lz4f.compress(cloudpickle.dumps(processor_instance), compression_level=pi_compression)
    closure = partial(
        _work_function,
        savemetrics=savemetrics,
        mmap=mmap,
        schema=schema,
        cachestrategy=cachestrategy,
        skipbadfiles=skipbadfiles,
        retries=retries,
        xrootdtimeout=xrootdtimeout,
        use_dataframes=use_dataframes,
        format="parquet"
    )
    # hack around dask/dask#5503 which is really a silly request but here we are
    if executor is dask_executor:
        executor_args['heavy_input'] = pi_to_send
        closure = partial(closure, processor_instance='heavy')
    else:
        closure = partial(closure, processor_instance=pi_to_send)

    exe_args = {
        'unit': 'chunk',
        'function_name': type(processor_instance).__name__,
        'use_dataframes': use_dataframes
    }
    exe_args.update(executor_args)
    wrapped_out = executor(chunks, closure, None, **exe_args)

    processor_instance.postprocess(wrapped_out["out"])
    if savemetrics and not use_dataframes:
        wrapped_out['metrics']['chunks'] = len(chunks)
        return wrapped_out["out"], wrapped_out['metrics']
    return wrapped_out['out']
