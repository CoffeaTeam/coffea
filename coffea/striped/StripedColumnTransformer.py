# here we want to define a group of columns from striped
# that all pertain to the same object type
# i.e. group = PhysicalColumnGroup(job,p4="Electron_p4",""NElectrons","Electron_p4","Electron_id1",...)
# this is specialized to striped so far
# we should try to peel it apart for other things


class ColumnGroup(object):
    def __init__(self, events, objName, *args):
        self._map = {}
        eventObj = getattr(events, objName)
        self._counts = getattr(events, objName).count
        for arg in args:
            callStack = arg.split('.')
            retval = getattr(eventObj, callStack[0])
            for i in range(1, len(callStack)):
                retval = getattr(retval, callStack[i])
            self._map[arg] = retval

    def __getitem__(self, name):
        return self._map[name]

    def columnsWithout(self, toremove):
        out = {}
        out.update(self._map)
        if isinstance(toremove, str):
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
    def __init__(self, events, objName, *args, **kwargs):
        self._hasp4 = False
        argkeys = kwargs.keys()
        self._p4 = None
        if 'p4' in argkeys:
            self._hasp4 = True
            self._p4 = {'p4': kwargs['p4']}
        elif 'pt' in argkeys and 'eta' in argkeys and 'phi' in argkeys and 'mass' in argkeys:
            self._hasp4 = True
            self._p4 = {key: kwargs[key] for key in ['pt', 'eta', 'phi', 'mass']}
        elif 'pt' in argkeys and 'eta' in argkeys and 'phi' in argkeys and 'energy' in argkeys:
            self._hasp4 = True
            self._p4 = {key: kwargs[key] for key in ['pt', 'eta', 'phi', 'energy']}
        elif 'px' in argkeys and 'py' in argkeys and 'pz' in argkeys and 'mass' in argkeys:
            self._hasp4 = True
            self._p4 = {key: kwargs[key] for key in ['px', 'py', 'pz', 'mass']}
        elif 'pt' in argkeys and 'phi' in argkeys and 'pz' in argkeys and 'energy' in argkeys:
            self._hasp4 = True
            self._p4 = {key: kwargs[key] for key in ['pt', 'phi', 'pz', 'energy']}
        elif 'px' in argkeys and 'py' in argkeys and 'pz' in argkeys and 'energy' in argkeys:
            self._hasp4 = True
            self._p4 = {key: kwargs[key] for key in ['px', 'py', 'pz', 'energy']}
        elif 'p' in argkeys and 'theta' in argkeys and 'phi' in argkeys and 'energy' in argkeys:
            self._hasp4 = True
            self._p4 = {key: kwargs[key] for key in ['p', 'theta', 'phi', 'energy']}
        elif 'p3' in argkeys and 'energy' in argkeys:
            self._hasp4 = True
            self._p4 = {key: kwargs[key] for key in ['p3', 'energy']}
        else:
            raise Exception('No valid definition of four-momentum found to build JaggedCandidateArray')
        allargs = list(args) + [val for val in kwargs.values()]
        super(PhysicalColumnGroup, self).__init__(events, objName, *allargs)

    def p4Name(self):
        if self._p4 is None:
            raise Exception('p4 is not set for this PhysicalColumnGroup')
        return self._p4.values()

    def p4Columns(self):
        return {key: self.columns()[value] for key, value in self._p4.items()}

    def otherColumns(self):
        return self.columnsWithout(self.p4Name())
