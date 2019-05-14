from fnal_column_analysis_tools import hist, processor
from copy import deepcopy
from concurrent.futures import as_completed

from tqdm import tqdm
import cloudpickle as cpkl
import lz4.frame as lz4f
import numpy as np
import pandas as pd

from parsl.app.app import python_app

lz4_clevel = 1


@python_app
def coffea_pyapp(dataset, fn, treename, chunksize, index, procstr):
    import uproot
    import cloudpickle as cpkl
    import lz4.frame as lz4f
    from fnal_column_analysis_tools import hist, processor
    from fnal_column_analysis_tools.processor.accumulator import accumulator

    lz4_clevel = 1

    # instrument xrootd source
    if not hasattr(uproot.source.xrootd.XRootDSource, '_read_real'):

        def _read(self, chunkindex):
            self.bytesread = getattr(self, 'bytesread', 0) + self._chunkbytes
            return self._read_real(chunkindex)

        uproot.source.xrootd.XRootDSource._read_real = uproot.source.xrootd.XRootDSource._read
        uproot.source.xrootd.XRootDSource._read = _read

    processor_instance = cpkl.loads(lz4f.decompress(procstr))

    file = uproot.open(fn)
    tree = file[treename]

    df = processor.LazyDataFrame(tree, chunksize, index)
    df['dataset'] = dataset

    vals = processor_instance.process(df)
    vals['_bytesread'] = accumulator(file.source.bytesread if isinstance(file.source, uproot.source.xrootd.XRootDSource) else 0)
    valsblob = lz4f.compress(cpkl.dumps(vals), compression_level=lz4_clevel)

    return valsblob


class ParslExecutor(object):

    def __init__(self):
        pass

    def __call__(self, dfk, items, processor_instance, output, unit='items', desc='Processing'):
        procstr = lz4f.compress(cpkl.dumps(processor_instance))

        nitems = len(items)
        ftr_to_item = set()
        for dataset, fn, treename, chunksize, index in items:
            ftr_to_item.add(coffea_pyapp(dataset, fn, treename, chunksize, index, procstr))

        for ftr in tqdm(as_completed(ftr_to_item), total=nitems, unit='items', desc='Processing'):
            blob = ftr.result()
            ftrhist = cpkl.loads(lz4f.decompress(blob))
            output.add(ftrhist)


parsl_executor = ParslExecutor()
