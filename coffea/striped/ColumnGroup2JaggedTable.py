from fnal_column_analysis_tools.analysis_objects.JaggedCandidateArray import JaggedCandidateArray
from fnal_column_analysis_tools.striped.StripedColumnTransformer import PhysicalColumnGroup
from fnal_column_analysis_tools.util import awkward


def jaggedFromColumnGroup(cgroup):
    if isinstance(cgroup, PhysicalColumnGroup):
        theargs = {}
        theargs.update(cgroup.p4Columns())
        theargs.update(cgroup.otherColumns())
        return JaggedCandidateArray.candidatesfromcounts(counts=cgroup.counts(),
                                                         **theargs)
    else:
        return awkward.JaggedArray.fromcounts(cgroup.counts(),
                                              awkward.Table(cgroup.columns()))
