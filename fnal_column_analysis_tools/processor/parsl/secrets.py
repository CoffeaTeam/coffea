from collections import OrderedDict
from concurrent.futures import as_completed
from parsl.app.app import python_app


def _parsl_work_function():
    raise NotImplementedError


@python_app
def derive_chunks(filename, treename, chunksize):
    import uproot
    nentries = uproot.numentries(fn, treename)
    return [(filename,chunksize,index) for index in range(nentries//chunksize + 1)]

@lru_cache(maxsize=128)
def _parsl_get_chunking(filelist, treename, chunksize):
    fn_to_index = { fn : idx for idx,fn in enumerate(filelist) } 
    future_to_fn = { derive_chunks(fn, treename, chunksize) : fn for fn in filelist}
    
    temp = [ 0 for fn in filelist ]
    for ftr in as_completed(future_to_fn):
        temp[fn_to_index[future_to_fn[ftr]]] = ftr.result()

    items = []
    for idx in range(len(temp)):
        items.extend(temp[idx))

    return items
