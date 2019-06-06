from __future__ import print_function, division

from coffea.lumi_tools import LumiData, LumiMask
from coffea.util import numpy as np

def test_lumidata():
    lumidata = LumiData("tests/samples/lumi_small.csv")
    
    runslumis = np.zeros((10,2), dtype=np.uint32)
    runslumis[:, 0] = lumidata._lumidata[0:10, 0]
    runslumis[:, 1] = lumidata._lumidata[0:10, 1]
    l = lumidata.get_lumi(runslumis)
    diff = abs(l - 1.539941814)
    print("lumi:", l, "diff:", diff)
    assert(diff < 0.1)

def test_lumimask():
    lumimask = LumiMask("tests/samples/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON.txt")
    runs = np.array([303825, 123], dtype=np.uint32)
    lumis = np.array([115, 123], dtype=np.uint32)
    mask = lumimask(runs, lumis)
    print("mask:", mask)
    assert(mask[0] == True)
    assert(mask[1] == False)
