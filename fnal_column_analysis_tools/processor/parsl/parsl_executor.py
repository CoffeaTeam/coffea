from fnal_column_analysis_tools import hist, processor
from copy import deepcopy
from concurrent.futures import as_completed
from collections.abc import Sequence

from tqdm import tqdm
import cloudpickle as cpkl
import lz4.frame as lz4f
import numpy as np
import pandas as pd
from collections import defaultdict

from parsl.app.app import python_app


@python_app
def coffea_pyapp(dataset, fn, treename, chunksize, index, procstr):
    import uproot
    import cloudpickle as cpkl
    import lz4.frame as lz4f
    from fnal_column_analysis_tools import hist, processor
    from fnal_column_analysis_tools.processor.accumulator import accumulator

    # instrument xrootd source
    if not hasattr(uproot.source.xrootd.XRootDSource, '_read_real'):

        def _read(self, chunkindex):
            self.bytesread = getattr(self, 'bytesread', 0) + self._chunkbytes
            return self._read_real(chunkindex)

        uproot.source.xrootd.XRootDSource._read_real = uproot.source.xrootd.XRootDSource._read
        uproot.source.xrootd.XRootDSource._read = _read

    processor_instance = cpkl.loads(lz4f.decompress(procstr))

    afile = uproot.open(fn)
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

    return vals, tree.numentries, dataset


class ParslExecutor(object):

    def __init__(self):
        self._counts = {}

    @property
    def counts(self):
        return self._counts

    def __call__(self, dfk, items, processor_instance, output, unit='items', desc='Processing'):
        procstr = lz4f.compress(cpkl.dumps(processor_instance))

        nitems = len(items)
        ftr_to_item = set()
        for dataset, fn, treename, chunksize, index in items:
            if dataset not in self._counts:
                self._counts[dataset] = 0
            ftr_to_item.add(coffea_pyapp(dataset, fn, treename, chunksize, index, procstr))

        for ftr in tqdm(as_completed(ftr_to_item), total=nitems, unit='items', desc='Processing'):
            ftrhist, nentries, dataset = ftr.result()
            self._counts[dataset] += nentries
            output.add(ftrhist)


parsl_executor = ParslExecutor()
