import awkward
import uproot_methods

JaggedWithLorentz = awkward.Methods.mixin(uproot_methods.classes.TLorentzVector.ArrayMethods, awkward.JaggedArray)

class JaggedDecoratedFourVector(awkward.JaggedArray,):
    def __init__(self,jagged):        
        super(JaggedDecoratedFourVector, self).__init__(jagged.starts,
                                                        jagged.stops,
                                                        jagged.content)
        if 'p4' not in self.columns:
            raise Exception('JaggedDecoratedFourVector declared without "p4" column: {}'.format(self.columns))
        
        self._ispair = False
        self._iscross = False
        if hasattr(jagged,'_ispair'):
            self._ispair = jagged._ispair
        if hasattr(jagged,'_iscross'):
            self._iscross = jagged._iscross
        
    @classmethod
    def fromcounts(cls,counts,p4,**kwargs):
        the_p4 = p4
        if not isinstance(p4,uproot_methods.TLorentzVectorArray):
            the_p4 = uproot_methods.TLorentzVectorArray(p4[:,0],p4[:,1],p4[:,2],p4[:,3])
        items = {'p4':the_p4}
        items.update(kwargs)
        return JaggedDecoratedFourVector(awkward.JaggedArray.fromcounts(counts,awkward.Table(items)))
    
    @property
    def p4(self):
        return self['p4']
    
    def at(self,what):
        raw = super(JaggedDecoratedFourVector,self).at(what)
        if 'p4' in raw.columns:
            return JaggedDecoratedFourVector(raw)
        return raw
    
    def distincts(self):
        return self.pairs(same=False)
    
    def pairs(self, same=True):
        outs = super(JaggedDecoratedFourVector, self).pairs(same)        
        if( sum(outs.counts) > 0 ):
            outs['p4'] = outs.at(0)['p4'] + outs.at(1)['p4']
        else:
            outs['p4'] = JaggedWithLorentz.fromcounts(outs.counts,[])        
        outs._ispair = True
        return JaggedDecoratedFourVector(outs)
    
    def cross(self, other):
        outs = super(JaggedDecoratedFourVector, self).cross(other)
        #currently JaggedArray.cross() has some funny behavior when it encounters the
        # p4 column and make some wierd new column... for now I just delete it and reorder
        # everything looks ok after that
        if outs._iscross:
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
        else:
            outs['p4'] = outs.at(0)['p4'] + outs.at(1)['p4']
            outs._iscross = True
        return JaggedDecoratedFourVector(outs)
    
    def __getattr__(self,what):
        if what in self.columns:
            return self[what]
        return getattr(super(JaggedDecoratedFourVector,self),what)
