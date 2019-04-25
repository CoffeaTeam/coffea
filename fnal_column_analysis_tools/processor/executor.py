import concurrent.futures
import time
from tqdm import tqdm
import uproot
from . import ProcessorABC, LazyDataFrame
from .accumulator import accumulator

try:
    from collections.abc import Mapping
    from functools import lru_cache
except ImportError:
    from collections import Mapping
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


def iterative_executor(items, function, accumulator, status=True):
    for i, item in tqdm(enumerate(items), disable=not status, unit='items', total=len(items), desc='Processing'):
        accumulator += function(item)
    return accumulator


def futures_executor(items, function, accumulator, workers=2, status=True):
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = set()
        try:
            futures.update(executor.submit(function, item) for item in items)
            with tqdm(disable=not status, unit='items', total=len(futures), desc='Processing') as pbar:
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
    file = uproot.open(fn)
    tree = file[treename]
    df = LazyDataFrame(tree, chunksize, index)
    df['dataset'] = dataset
    out = processor_instance.process(df)
    out['_bytesread'] = accumulator(file.source.bytesread if isinstance(file.source, uproot.source.xrootd.XRootDSource) else 0)
    return out


@lru_cache(maxsize=128)
def _get_chunking(filelist, treename, chunksize):
    items = []
    for fn in filelist:
        nentries = uproot.numentries(fn, treename)
        for index in range(nentries//chunksize + 1):
            items.append((fn, chunksize, index))
    return items


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
    for dataset, filelist in tqdm(fileset.items(), desc='Preprocessing'):
        for chunk in _get_chunking(tuple(filelist), treename, chunksize):
            items.append((dataset, chunk[0], treename, chunk[1], chunk[2], processor_instance))

    output = processor_instance.accumulator.identity()
    executor(items, _work_function, output, **executor_args)
    processor_instance.postprocess(output)
    return output
