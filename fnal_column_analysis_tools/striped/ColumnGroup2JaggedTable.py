from fnal_column_analysis_tools.analysis_objects.JaggedCandidateArray import JaggedCandidateArray
from fnal_column_analysis_tools.striped.StripedColumnTransformer import PhysicalColumnGroup
import awkward

def jaggedFromColumnGroup(cgroup):
    if isinstance(cgroup,PhysicalColumnGroup):
        return JaggedCandidateArray.candidatesfromcounts(counts = cgroup.counts(),
                                                         p4 = cgroup.p4Column(),
                                                         **cgroup.otherColumns())
    else:
        return awkward.JaggedArray.fromcounts(cgroup.counts(),
                                              awkward.Table(cgroup.columns()))
                                                    
