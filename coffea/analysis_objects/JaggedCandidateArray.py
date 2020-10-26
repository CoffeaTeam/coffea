from coffea.analysis_objects.JaggedCandidateMethods import JaggedCandidateMethods
from coffea.util import awkward


class JaggedCandidateArray(JaggedCandidateMethods, awkward.JaggedArray):
    """Candidate methods mixed in with an awkward0 JaggedArray"""

    pass
