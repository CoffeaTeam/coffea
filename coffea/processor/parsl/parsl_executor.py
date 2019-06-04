from coffea import hist, processor
from copy import deepcopy
from concurrent.futures import as_completed
from collections.abc import Sequence

from tqdm import tqdm
import cloudpickle as cpkl
import pickle as pkl
import lz4.frame as lz4f
import numpy as np
import pandas as pd

from parsl.app.app import python_app
from .timeout import timeout

lz4_clevel = 1


@python_app
@timeout
def coffea_pyapp(dataset, fn, treename, chunksize, index, procstr, timeout=None):
    import uproot
    import cloudpickle as cpkl
    import pickle as pkl
    import lz4.frame as lz4f
    from coffea import hist, processor
    from coffea.processor.accumulator import accumulator
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    uproot.XRootDSource.defaults["parallel"] = False

    lz4_clevel = 1

    # instrument xrootd source
    if not hasattr(uproot.source.xrootd.XRootDSource, '_read_real'):

        def _read(self, chunkindex):
            self.bytesread = getattr(self, 'bytesread', 0) + self._chunkbytes
            return self._read_real(chunkindex)

        uproot.source.xrootd.XRootDSource._read_real = uproot.source.xrootd.XRootDSource._read
        uproot.source.xrootd.XRootDSource._read = _read

    processor_instance = cpkl.loads(lz4f.decompress(procstr))

    afile = None
    for i in range(5):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(uproot.open, fn)
            try:
                afile = future.result(timeout=5)
            except TimeoutError:
                afile = None
            else:
                break

    if afile is None:
        raise Exception('unable to open: %s' % fn)
    tree = None
    if isinstance(treename, str):
        tree = afile[treename]
    elif isinstance(treename, Sequence):
        for name in reversed(treename):
            if name in afile:
                tree = afile[name]
    else:
        raise Exception('treename must be a str or Sequence but is a %s!' % repr(type(treename)))

    if tree is None:
        raise Exception('No tree found, out of possible tree names: %s' % repr(treename))

    df = processor.LazyDataFrame(tree, chunksize, index, flatten=True)
    df['dataset'] = dataset

    vals = processor_instance.process(df)
    vals['_bytesread'] = accumulator(afile.source.bytesread if isinstance(afile.source, uproot.source.xrootd.XRootDSource) else 0)
    valsblob = lz4f.compress(pkl.dumps(vals), compression_level=lz4_clevel)

    istart = chunksize * index
    istop = min(tree.numentries, (index + 1) * chunksize)
    return valsblob, (istop - istart), dataset


class ParslExecutor(object):

    def __init__(self):
        self._counts = {}

    @property
    def counts(self):
        return self._counts

    def __call__(self, dfk, items, processor_instance, output, unit='items', desc='Processing', timeout=None):
        procstr = lz4f.compress(cpkl.dumps(processor_instance))

        nitems = len(items)
        ftr_to_item = set()
        for dataset, fn, treename, chunksize, index in items:
            if dataset not in self._counts:
                self._counts[dataset] = 0
            ftr_to_item.add(coffea_pyapp(dataset, fn, treename, chunksize, index, procstr, timeout=timeout))

        for ftr in tqdm(as_completed(ftr_to_item), total=nitems, unit='items', desc='Processing'):
            blob, nentries, dataset = ftr.result()
            self._counts[dataset] += nentries
            output.add(pkl.loads(lz4f.decompress(blob)))


parsl_executor = ParslExecutor()
