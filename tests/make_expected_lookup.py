import numpy as np
import ROOT

from dummy_distributions import dummy_pt_eta

counts, test_in1, test_in2 = dummy_pt_eta()

f = ROOT.TFile.Open("samples/testSF2d.root")
sf = f.Get("scalefactors_Tight_Electron")

test_out = np.empty_like(test_in1)
for i, (eta,pt) in enumerate(zip(test_in1, test_in2)):
    ib = sf.FindBin(eta, pt)
    test_out[i] = sf.GetBinContent(ib)

print repr(test_out)
