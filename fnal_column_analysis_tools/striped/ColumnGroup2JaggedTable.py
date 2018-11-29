from ..analysis_objects.JaggedDecoratedFourVector import JaggedWithLorentz, JaggedDecoratedFourVector
import awkward

def jaggedFromColumnGroup(cgroup):
    if isinstance(cgroup,PhysicalColumnGroup):
        return JaggedDecoratedFourVector.fromcounts(counts = cgroup.counts(),
                                                    p4 = cgroup.p4Column(),
                                                    **cgroup.otherColumns())
    else:
        return awkward.JaggedArray.fromcounts(cgroup.counts(),
                                              awkward.Table(cgroup.columns()))
                                                    
