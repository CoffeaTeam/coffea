import concurrent.futures
import time
from tqdm import tqdm
import uproot
from . import ProcessorABC, LazyDataFrame

try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping


def iterative_executor(items, function, accumulator, status=True):
    for i, item in tqdm(enumerate(items), disable=not status, unit='items', total=len(items)):
        accumulator += function(item)
    return accumulator


def futures_executor(items, function, accumulator, workers=2, status=True):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = set()
        try:
            futures.update(executor.submit(function, item) for item in items)
            with tqdm(disable=not status, unit='items', total=len(futures)) as pbar:
                while len(futures) > 0:
                    finished = set(job for job in futures if job.done())
                    for job in finished:
                        accumulator += job.result()
                        pbar.update(1)
                    futures -= finished
                    del finished
                    time.sleep(1)
        except KeyboardInterrupt:
            for job in futures:
                job.cancel()
            if status:
                print("Received SIGINT, killed pending jobs.  Running jobs will continue to completion.")
                print("Running jobs:", sum(1 for j in futures if j.running()))
        except Exception:
            for job in futures:
                job.cancel()
            raise
    return accumulator


def condor_executor(items, function, accumulator, workers, status=True):
    raise NotImplementedError


def _work_function(item):
    dataset, fn, treename, chunksize, index, processor_instance = item
    tree = uproot.open(fn)[treename]
    df = LazyDataFrame(tree, chunksize, index)
    df['dataset'] = dataset
    return processor_instance.process(df)


def run_uproot_job(fileset, treename, processor_instance, executor, executor_args={}, chunksize=500000):
    '''
    A convenience wrapper to submit jobs for a file set, which is a
    dictionary of dataset: [file list] entries.  Supports only uproot
    reading, via the LazyDataFrame class.  For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.
    fileset: dictionary {dataset: [file, file], }
    treename: name of tree inside each root file
    processor_instance: an instance of a class deriving from ProcessorABC
    executor: any of iterative_executor, futures_executor, etc.
                In general, a function that takes 3 arguments: items, function accumulator
                and performs some action equivalent to:
                for item in items: accumulator += function(item)
    executor_args: extra arguments to pass to executor
    chunksize: number of entries to process at a time in the data frame
    '''
    if not isinstance(fileset, Mapping):
        raise ValueError("Expected fileset to be a mapping dataset: list(files)")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")

    items = []
    for dataset, filelist in fileset.items():
        for fn in filelist:
            file = uproot.open(fn)
            tree = file[treename]
            for index in range(tree.numentries//chunksize + 1):
                items.append((dataset, fn, treename, chunksize, index, processor_instance))

    accumulator = processor_instance.accumulator.identity()
    executor(items, _work_function, accumulator, **executor_args)
    processor_instance.postprocess(accumulator)
    return accumulator
