import awkward
import uproot_methods

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
        items.update(kwargs)
        return cls.fromcounts(counts,awkward.Table(items))
    
    @property
    def p4(self):
        return self['p4']
    
    def at(self,what):
        thewhat = super(JaggedCandidateMethods,self).at(what)
        if 'p4' in thewhat.columns:
            return self.fromjagged(thewhat)
        return thewhat
    
    def distincts(self):
        return self.pairs(same=False)
    
    def pairs(self, same=True):
        outs = super(JaggedCandidateMethods, self).pairs(same)
        outs['p4'] = outs.at(0)['p4'] + outs.at(1)['p4']
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
        return self.fromjagged(outs)

    def __getattr__(self,what):
        if what in self.columns:
            return self[what]
        return getattr(super(JaggedCandidateMethods,self),what)

JaggedCandidateArray = awkward.Methods.mixin(JaggedCandidateMethods, awkward.JaggedArray)
