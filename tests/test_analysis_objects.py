from __future__ import print_function, division

import fnal_column_analysis_tools
from fnal_column_analysis_tools.analysis_objects import JaggedCandidateArray, JaggedTLorentzVectorArray
import uproot
import uproot_methods
import awkward
import numpy as np

from dummy_distributions import dummy_four_momenta

def test_analysis_objects():
    counts, px, py, pz, energy = dummy_four_momenta()
    thep4 = np.stack((px,py,pz,energy)).T
    
    #test JaggedTLorentzVectorArray
    tlva1 = uproot_methods.TLorentzVectorArray(px,py,pz,energy)
    tlva2 = uproot_methods.TLorentzVectorArray(thep4[:,0],thep4[:,1],
                                               thep4[:,2],thep4[:,3])
    jtlva1 = JaggedTLorentzVectorArray.fromcounts(counts,tlva1)
    jtlva2 = JaggedTLorentzVectorArray.fromcounts(counts,tlva2)
    
    jtlva1_selection1 = jtlva1[jtlva1.counts > 0]
    jtlva1_selection2 = jtlva1_selection1[jtlva1_selection1.pt > 5]

    jtlva2_selection1 = jtlva2[jtlva2.counts > 0]
    jtlva2_selection2 = jtlva1_selection1[jtlva2_selection1.pt > 5]
    
    diffx = np.abs(jtlva1.x - jtlva2.x)
    diffy = np.abs(jtlva1.y - jtlva2.y)
    diffz = np.abs(jtlva1.z - jtlva2.z)
    difft = np.abs(jtlva1.t - jtlva2.t)
    assert (diffx < 1e-8).flatten().all()
    assert (diffy < 1e-8).flatten().all()
    assert (diffz < 1e-8).flatten().all()
    assert (difft < 1e-8).flatten().all()

    #test JaggedCandidateArray
    jca1 = JaggedCandidateArray.candidatesfromcounts(counts,p4=thep4)
    jca2 = JaggedCandidateArray.candidatesfromcounts(counts,p4=thep4)
    assert( (jca1.offsets == jca2.offsets).all() )

    addon1 = jca1.zeros_like()
    addon2 = jca2.ones_like()
    jca1['addon'] = addon1
    jca2['addon'] = addon2

    jca1.add_attributes(addonFlat=addon1.flatten(),addonJagged=addon1)
    
    diffm = np.abs(jca1.p4.mass - jca2.p4.mass)
    assert( (jca1.offsets == jca2.offsets).all() )
    diffpt = np.abs(jca1.p4.pt - jca2.p4.pt)
    assert( (jca1.offsets == jca2.offsets).all() )
    eta2 = jca2.p4.eta
    eta1 = jca1.p4.eta
    print (np.sum(eta1.counts),np.sum(eta2.counts))
    diffeta_temp = np.abs(eta1 - eta2)
    diffeta = np.abs(jca1.p4.eta - jca2.p4.eta)
    assert( (jca1.offsets == jca2.offsets).all() )
    assert (diffm < 1e-8).flatten().all()
    assert (diffpt < 1e-8).flatten().all()
    assert (diffeta < 1e-8).flatten().all()
    
    #test fast functions
    fastfs = ['pt','eta','phi','mass']
    for func in fastfs:
        func1 = getattr(jca1,func)        
        func2 = getattr(jca1.p4,func)
        dfunc = np.abs(func1 - func2)
        assert (dfunc < 1e-8).flatten().all()

    adistinct = jca1.distincts()
    apair = jca1.pairs()
    across = jca1.cross(jca2)
    
    assert 'p4' in adistinct.columns
    assert 'p4' in apair.columns
    assert 'p4' in across.columns
    
    admsum = (adistinct.i0.p4 + adistinct.i1.p4).mass
    apmsum = (apair.i0.p4 + apair.i1.p4).mass
    acmsum = (across.i0.p4 + across.i1.p4).mass
    diffadm = np.abs(adistinct.p4.mass - admsum)
    diffapm = np.abs(apair.p4.mass - apmsum)
    diffacm = np.abs(across.p4.mass - acmsum)
    
    assert (diffadm < 1e-8).flatten().all()
    assert (diffapm < 1e-8).flatten().all()
    assert (diffacm < 1e-8).flatten().all()
    
    selection11 = jca1[jca1.counts > 0]
    selection12 = selection11[selection11.p4.pt > 5]
    
    selection21 = jca2[jca2.counts > 0]
    selection22 = selection21[selection21.p4.pt > 5]

    diffcnts = selection12.counts - jtlva1_selection2.counts
    diffm = np.abs(selection12.p4.mass - jtlva1_selection2.mass)
    diffaddon = selection12.addon - selection22.addon
    assert (diffcnts == 0).flatten().all()
    assert (diffm < 1e-8).flatten().all()
    assert (diffaddon == -1).flatten().all()


