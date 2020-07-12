from __future__ import print_function, division

import sys
import pytest
import coffea
from coffea.analysis_objects import JaggedCandidateArray, JaggedTLorentzVectorArray
import uproot
import uproot_methods
from coffea.util import awkward
from coffea.util import numpy as np

from dummy_distributions import dummy_four_momenta, gen_reco_TLV

if sys.platform.startswith("win"):
    pytest.skip("skipping tests that only function in linux", allow_module_level=True)

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
    acrossn = jca1.cross(jca2, nested=True)
    achoose2 = jca1.choose(2)
    achoose3 = jca1.choose(3)
    
    assert 'p4' in adistinct.columns
    assert 'p4' in apair.columns
    assert 'p4' in across.columns
    assert 'p4' in acrossn.columns
    assert 'p4' in achoose2.columns
    assert 'p4' in achoose3.columns
    
    admsum = (adistinct.i0.p4 + adistinct.i1.p4).mass
    apmsum = (apair.i0.p4 + apair.i1.p4).mass
    acmsum = (across.i0.p4 + across.i1.p4).mass
    ach3msum = (achoose3.i0.p4 + achoose3.i1.p4 + achoose3.i2.p4).mass
    diffadm = np.abs(adistinct.p4.mass - admsum)
    diffapm = np.abs(apair.p4.mass - apmsum)
    diffacm = np.abs(across.p4.mass - acmsum)
    diffachm = np.abs(achoose2.p4.mass - admsum)
    diffach3m = np.abs(achoose3.p4.mass - ach3msum)
    
    assert (diffadm < 1e-8).flatten().all()
    assert (diffapm < 1e-8).flatten().all()
    assert (diffacm < 1e-8).flatten().all()
    assert (diffachm < 1e-8).flatten().all()
    assert (diffach3m < 1e-8).flatten().all()

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

    #test gen-reco matching
    gen, reco = gen_reco_TLV()
    flat_gen = gen.flatten()
    gen_px,gen_py,gen_pz,gen_e = flat_gen.x,flat_gen.y,flat_gen.z,flat_gen.t
    flat_reco = reco.flatten()
    reco_px,reco_py,reco_pz,reco_e = flat_reco.x,flat_reco.y,flat_reco.z,flat_reco.t
    jca_gen = JaggedCandidateArray.candidatesfromcounts(gen.counts,
                                                        px=gen_px,py=gen_py,
                                                        pz=gen_pz,energy=gen_e)
    jca_reco = JaggedCandidateArray.candidatesfromcounts(reco.counts,
                                                         px=reco_px,py=reco_py,
                                                         pz=reco_pz,energy=reco_e)
    jca_reco_pt = JaggedCandidateArray.candidatesfromcounts(reco.counts,
                                                            pt=jca_reco.pt.content,eta=jca_reco.eta.content,
                                                            phi=jca_reco.phi.content,mass=jca_reco.mass.content)
    print('gen eta: ', jca_gen.p4.eta,'\n gen phi:', jca_gen.p4.phi)
    print('reco eta: ', jca_reco.p4.eta,'\n reco phi:', jca_reco.p4.phi)
    match_mask = jca_reco.match(jca_gen, deltaRCut=0.3)
    print('match mask: ',  match_mask)
    fast_match_mask = jca_reco.fastmatch(jca_gen, deltaRCut=0.3)
    print('fastmatch mask: ', fast_match_mask)
    assert((match_mask == fast_match_mask).all().all())
    print('arg matches: ',jca_reco.argmatch(jca_gen,deltaRCut=0.3))
    argmatch_nocut = jca_gen.argmatch(jca_reco).flatten()
    argmatch_dr03  = jca_gen.argmatch(jca_reco, deltaRCut=0.3).flatten()
    argmatch_dr03_dpt01 = jca_gen.argmatch(jca_reco, deltaRCut=0.3, deltaPtCut=0.1).flatten()
    assert (argmatch_nocut.size==5)
    assert (argmatch_dr03[argmatch_dr03 != -1].size==3)
    assert (argmatch_dr03_dpt01[argmatch_dr03_dpt01 != -1].size==2)
    assert (jca_gen.match(jca_reco,deltaRCut=0.3).flatten().flatten().sum()==3)
    assert (jca_gen.match(jca_reco,deltaRCut=0.3,deltaPtCut=0.1).flatten().flatten().sum()==2)

    # test various four-momentum constructors
    ptetaphiE_test = JaggedCandidateArray.candidatesfromcounts(jca_reco.counts,
                                                               pt=jca_reco.pt,
                                                               eta=jca_reco.eta,
                                                               phi=jca_reco.phi,
                                                               energy=jca_reco.p4.energy)

    pxpypzM_test = JaggedCandidateArray.candidatesfromcounts(jca_reco.counts,
                                                             px=jca_reco.p4.x,
                                                             py=jca_reco.p4.y,
                                                             pz=jca_reco.p4.z,
                                                             mass=jca_reco.mass)

    ptphipzE_test = JaggedCandidateArray.candidatesfromcounts(jca_reco.counts,
                                                              pt=jca_reco.pt,
                                                              phi=jca_reco.phi,
                                                              pz=jca_reco.p4.z,
                                                              energy=jca_reco.p4.energy)

    pthetaphiE_test = JaggedCandidateArray.candidatesfromcounts(jca_reco.counts,
                                                                p=jca_reco.p4.p,
                                                                theta=jca_reco.p4.theta,
                                                                phi=jca_reco.phi,
                                                                energy=jca_reco.p4.energy)

    p4cart_test = JaggedCandidateArray.candidatesfromcounts(jca_reco.counts,
                                                            p4=jca_reco.p4)

    p4ptetaphiM_test = JaggedCandidateArray.candidatesfromcounts(jca_reco.counts,
                                                                 p4=jca_reco_pt.p4)
