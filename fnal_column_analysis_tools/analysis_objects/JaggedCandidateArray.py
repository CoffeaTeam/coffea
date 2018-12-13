import awkward
import uproot_methods
import numpy as np

JaggedTLorentzVectorArray = awkward.Methods.mixin(uproot_methods.classes.TLorentzVector.ArrayMethods, awkward.JaggedArray)

#functions to quickly cash useful quantities
def fast_pt(p4):
    return np.hypot(p4.x,p4.y)

def fast_eta(p4):
    px = p4.x
    py = p4.y
    pz = p4.z
    p3mag = np.sqrt(px*px + py*py + pz*pz)
    return np.arctanh(pz/p3mag)

def fast_phi(p4):
    return np.arctan2(p4.y,p4.x)

def fast_mass(p4):
    px = p4.x
    py = p4.y
    pz = p4.z
    en = p4.t
    p3mag2 = (px*px + py*py + pz*pz)
    return np.sqrt(np.abs(en*en - p3mag2))

class JaggedCandidateMethods(awkward.Methods):
    
    @classmethod
    def candidatesfromcounts(cls,counts,p4,**kwargs):
        offsets = awkward.array.jagged.counts2offsets(counts)
        return cls.candidatesfromoffsets(offsets,p4,**kwargs)
    
    @classmethod
    def candidatesfromoffsets(cls,offsets,p4,**kwargs):
        items = {}
        if isinstance(p4,uproot_methods.TLorentzVectorArray):
            items['p4'] = p4
        else:
            items['p4'] = uproot_methods.TLorentzVectorArray(p4[:,0],p4[:,1],
                                                             p4[:,2],p4[:,3])
        thep4 = items['p4']
        items['__fast_pt'] = fast_pt(thep4)
        items['__fast_eta'] = fast_eta(thep4)
        items['__fast_phi'] = fast_phi(thep4)
        items['__fast_mass'] = fast_mass(thep4)
        items.update(kwargs)
        return cls.fromoffsets(offsets,awkward.Table(items))
    
    @property
    def p4(self):
        return self['p4']
    
    @property
    def pt(self):
        return self['__fast_pt']
    
    @property
    def eta(self):
        return self['__fast_eta']
    
    @property
    def phi(self):
        return self['__fast_phi']
    
    @property
    def mass(self):
        return self['__fast_mass']
    
    @property
    def i0(self):
        if 'p4' in self['0'].columns: return self.fromjagged(self['0'])
        return self['0']
    
    @property
    def i1(self):
        if 'p4' in self['1'].columns: return self.fromjagged(self['1'])
        return self['1']

    @property
    def i2(self):
        if 'p4' in self['2'].columns: return self.fromjagged(self['2'])
        return self['2']

    @property
    def i3(self):
        if 'p4' in self['3'].columns: return self.fromjagged(self['3'])
        return self['3']

    @property
    def i4(self):
        if 'p4' in self['4'].columns: return self.fromjagged(self['4'])
        return self['4']
    
    @property
    def i5(self):
        if 'p4' in self['5'].columns: return self.fromjagged(self['5'])
        return self['5']

    @property
    def i6(self):
        if 'p4' in self['6'].columns: return self.fromjagged(self['6'])
        return self['6']
    
    @property
    def i7(self):
        if 'p4' in self['7'].columns: return self.fromjagged(self['7'])
        return self['7']

    @property
    def i8(self):
        if 'p4' in self['8'].columns: return self.fromjagged(self['8'])
        return self['8']
    
    @property
    def i9(self):
        if 'p4' in self['9'].columns: return self.fromjagged(self['9'])
        return self['9']

    def distincts(self, nested=False):
        outs = super(JaggedCandidateMethods, self).distincts(nested)
        outs['p4'] = outs.i0['p4'] + outs.i1['p4']
        thep4 = outs['p4']
        outs['__fast_pt'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_pt(thep4.content))
        outs['__fast_eta'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_eta(thep4.content))
        outs['__fast_phi'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_phi(thep4.content))
        outs['__fast_mass'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_mass(thep4.content))
        return self.fromjagged(outs)

    def pairs(self, nested=False):
        outs = super(JaggedCandidateMethods, self).pairs(nested)
        outs['p4'] = outs.i0['p4'] + outs.i1['p4']
        thep4 = outs['p4']
        outs['__fast_pt'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_pt(thep4.content))
        outs['__fast_eta'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_eta(thep4.content))
        outs['__fast_phi'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_phi(thep4.content))
        outs['__fast_mass'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_mass(thep4.content))
        return self.fromjagged(outs)
    
    def cross(self, other, nested=False):
        outs = super(JaggedCandidateMethods, self).cross(other,nested)
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
                outs['p4'] = outs[key]['p4']
            else:
                outs['p4'] = outs['p4'] + outs[key]['p4']
        thep4 = outs['p4']
        outs['__fast_pt'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_pt(thep4.content))
        outs['__fast_eta'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_eta(thep4.content))
        outs['__fast_phi'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_phi(thep4.content))
        outs['__fast_mass'] = awkward.JaggedArray.fromoffsets(outs.offsets,fast_mass(thep4.content))
        return self.fromjagged(outs)
        
    def __getattr__(self,what):
        if what in self.columns:
            return self[what]
        thewhat = getattr(super(JaggedCandidateMethods,self),what)
        if 'p4' in thewhat.columns: return self.fromjagged(thewhat)
        return thewhat

JaggedCandidateArray = awkward.Methods.mixin(JaggedCandidateMethods, awkward.JaggedArray)
