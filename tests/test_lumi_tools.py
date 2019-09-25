from __future__ import print_function, division

from coffea.lumi_tools import LumiData, LumiMask, LumiList
from coffea.util import numpy as np

def test_lumidata():
    from numba import types
    from numba.typed import Dict
    
    lumidata = LumiData("tests/samples/lumi_small.csv")
    
    runslumis = np.zeros((10,2), dtype=np.uint32)
    runslumis[:, 0] = lumidata._lumidata[0:10, 0]
    runslumis[:, 1] = lumidata._lumidata[0:10, 1]
    l = lumidata.get_lumi(runslumis)
    diff = abs(l - 1.539941814)
    print("lumi:", l, "diff:", diff)
    assert(diff < 1e-4)

    # test build_lumi_table_kernel
    py_index = Dict.empty(
        key_type=types.Tuple([types.uint32, types.uint32]),
        value_type=types.float64
    )
    pyruns = lumidata._lumidata[:, 0].astype('u4')
    pylumis = lumidata._lumidata[:, 1].astype('u4')
    LumiData._build_lumi_table_kernel.py_func(pyruns, pylumis, lumidata._lumidata, py_index)

    assert(len(py_index) == len(lumidata.index))

    # test get_lumi_kernel
    py_tot_lumi = np.zeros((1, ), dtype=np.float64)
    LumiData._get_lumi_kernel.py_func(runslumis[:, 0], runslumis[:, 1], py_index, py_tot_lumi)

    assert(abs(py_tot_lumi[0] - l) < 1e-4)

def test_lumimask():
    lumimask = LumiMask("tests/samples/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt")
    runs = np.array([303825, 123], dtype=np.uint32)
    lumis = np.array([115, 123], dtype=np.uint32)
    mask = lumimask(runs, lumis)
    print("mask:", mask)
    assert(mask[0] == True)
    assert(mask[1] == False)

    # test underlying py_func
    py_mask = np.zeros(dtype='bool', shape=runs.shape)
    LumiMask._apply_run_lumi_mask_kernel.py_func(lumimask._masks,
                                                runs, lumis,
                                                py_mask)

    assert(np.all(mask == py_mask))

def test_lumilist():
    lumidata = LumiData("tests/samples/lumi_small.csv")
    
    runslumis1 = np.zeros((10,2), dtype=np.uint32)
    runslumis1[:, 0] = lumidata._lumidata[0:10, 0]
    runslumis1[:, 1] = lumidata._lumidata[0:10, 1]

    runslumis2 = np.zeros((10,2), dtype=np.uint32)
    runslumis2[:, 0] = lumidata._lumidata[10:20, 0]
    runslumis2[:, 1] = lumidata._lumidata[10:20, 1]

    llist1 = LumiList(runs=runslumis1[:, 0], lumis=runslumis1[:, 1])
    llist2 = LumiList(runs=runslumis2[:, 0], lumis=runslumis2[:, 1])
    llist3 = LumiList()

    llist3 += llist1
    llist3 += llist2

    lumi1 = lumidata.get_lumi(llist1)
    lumi2 = lumidata.get_lumi(llist2)
    lumi3 = lumidata.get_lumi(llist3)
    
    assert(abs(lumi3 - (lumi1 + lumi2)) < 1e-4)

    llist1.clear()
    assert(llist1.array.size == 0)
