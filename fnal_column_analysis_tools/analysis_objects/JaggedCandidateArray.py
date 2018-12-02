import awkward
import uproot_methods

import numpy as np

#functions to quickly cash useful quantities
def fast_pt(px,py):
    return np.hypot(px,py)

def fast_eta(px,py,pz):
    p3mag = np.sqrt(px*px + py*py + pz*pz)
    return np.arctanh(pz/p3mag)

def fast_mass(px,py,pz,E):
    return np.sqrt(E*E - (px*px + py*py + pz*pz))

def fast_phi(px,py):
    return np.arctan2(py,px)

JaggedTLorentzVectorArray = awkward.Methods.mixin(uproot_methods.classes.TLorentzVector.ArrayMethods, awkward.JaggedArray)

class JaggedCandidateMethods(awkward.Methods):
    
    @classmethod
    def candidatesfromcounts(cls,counts,p4,**kwargs):
        items = {}
        if isinstance(p4,uproot_methods.TLorentzVectorArray):
            items['p4'] = p4
        else:
            items['p4'] = uproot_methods.TLorentzVectorArray(p4[:,0],p4[:,1],
                                                             p4[:,2],p4[:,3])
        thep4 = items['p4']
        items['__fast_mass'] = fast_mass(thep4.x,thep4.y,thep4.z,thep4.t)
        items['__fast_pt'] = fast_pt(thep4.x,thep4.y)
        items['__fast_eta'] = fast_eta(thep4.x,thep4.y,thep4.z)
        items['__fast_phi'] = fast_phi(thep4.x,thep4.y)
        items.update(kwargs)
        return cls.fromcounts(counts,awkward.Table(items))
    
    @property
    def p4(self):
        return self['p4']
    
    @property
    def mass(self):
        return self['__fast_mass']

    @property
    def pt(self):
        return self['__fast_pt']

    @property
    def eta(self):
        return self['__fast_eta']

    @property
    def phi(self):
        return self['__fast_phi']

    def at(self,what):
        thewhat = super(JaggedCandidateMethods,self).at(what)
        if 'p4' in thewhat.columns:
            return self.fromjagged(thewhat)
        return thewhat
    
    def distincts(self):
        return self.pairs(same=False)
    
    def pairs(self, same=True):
        outs = super(JaggedCandidateMethods, self).pairs(same)
        thep4 = outs.at(0)['p4'] + outs.at(1)['p4']
        outs['p4'] = thep4
        outs['__fast_mass'] = awkward.JaggedArray.fromcounts(thep4.counts,fast_mass(thep4.x,thep4.y,thep4.z,thep4.t))
        outs['__fast_pt'] = awkward.JaggedArray.fromcounts(thep4.counts,fast_pt(thep4.x,thep4.y))
        outs['__fast_eta'] = awkward.JaggedArray.fromcounts(thep4.counts,fast_eta(thep4.x,thep4.y,thep4.z))
        outs['__fast_phi'] = awkward.JaggedArray.fromcounts(thep4.counts,fast_phi(thep4.x,thep4.y))
        return self.fromjagged(outs)
    
    def cross(self, other):
        outs = super(JaggedCandidateMethods, self).cross(other)
        #currently JaggedArray.cross() has some funny behavior when it encounters the
        # p4 column and makes some wierd new column... for now I just delete it and reorder
        # everything looks ok after that
        keys = outs.columns
        reorder = False
        for key in keys:
            if not isinstance(outs[key].content,awkward.array.table.Table):
                del outs[key]
                reorder = True
        if reorder:
            keys = outs.columns
            realkey = {}
            for i in xrange(len(keys)):
                realkey[keys[i]] = str(i)
            for key in keys:
                if realkey[key] != key:
                    outs[realkey[key]] = outs[key]
                    del outs[key]
        keys = outs.columns
        for key in keys:
            if 'p4' not in outs.columns:
                outs['p4'] = outs.at(int(key))['p4']
            else:
                outs['p4'] = outs['p4'] + outs.at(int(key))['p4']
        thep4 = outs['p4']
        p4content = thep4.content
        outs['__fast_mass'] = awkward.JaggedArray.fromcounts(thep4.counts,fast_mass(p4content.x,
                                                                                    p4content.y,
                                                                                    p4content.z,
                                                                                    p4content.t))
        outs['__fast_pt'] = awkward.JaggedArray.fromcounts(thep4.counts,fast_pt(p4content.x,
                                                                                p4content.y))
        outs['__fast_eta'] = awkward.JaggedArray.fromcounts(thep4.counts,fast_eta(p4content.x,
                                                                                  p4content.y,
                                                                                  p4content.z))
        outs['__fast_phi'] = awkward.JaggedArray.fromcounts(thep4.counts,fast_phi(p4content.x,
                                                                                  p4content.y))
        return self.fromjagged(outs)

    def __getattr__(self,what):
        if what in self.columns:
            return self[what]
        return getattr(super(JaggedCandidateMethods,self),what)

JaggedCandidateArray = awkward.Methods.mixin(JaggedCandidateMethods, awkward.JaggedArray)
