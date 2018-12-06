# here we want to define a group of columns from striped
# that all pertain to the same object type
# i.e. group = PhysicalColumnGroup(job,p4="Electron_p4",""NElectrons","Electron_p4","Electron_id1",...)
# this is specialized to striped so far
# we should try to peel it apart for other things

class ColumnGroup(object):
    def __init__(self,events,objName,*args):
        self._map = {}
        eventObj = getattr(events,objName)
        self._counts = getattr(events,objName).count
        for arg in args:
            callStack = arg.split('.')
            retval = getattr(eventObj,callStack[0])
            for i in range(1,len(callStack)):
                retval = getattr(retval,callStack[i])
            self._map[arg] = retval
            
    def __getitem__(self,name):
        return self._map[name]
    
    def columnsWithout(self,toremove):
        out = {}
        out.update(self._map)
        if isinstance(toremove,str):
            del out[toremove]
        else:
            for key in toremove:
                del out[key]
        return out
    
    def columns(self):
        return self._map
    
    def counts(self):
        return self._counts
    
class PhysicalColumnGroup(ColumnGroup):
    def __init__(self,events,objName,p4Name,*args):
        self._p4  = p4Name
        allargs = [p4Name]
        allargs.extend(args)        
        super(PhysicalColumnGroup,self).__init__(events,objName,*allargs)
        if p4Name is not None:
            self.setP4Name(p4Name)
    
    def setP4Name(self,name):
        if name not in self.columns().keys():
            raise Exception('{} not an available name in this PhysicalColumnGroup'.format(name))
        self._p4 = name
    
    def p4Name(self):
        if self._p4 is None:
            raise Exception('p4 is not set for this PhysicalColumnGroup')
        return self._p4
    
    def p4Column(self):        
        return self[self.p4Name()]
    
    def otherColumns(self):
        return self.columnsWithout(self.p4Name())

