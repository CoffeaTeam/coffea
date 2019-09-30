from .JaggedCandidateMethods import JaggedCandidateMethods
from ..util import awkward

JaggedCandidateArray = awkward.Methods.mixin(JaggedCandidateMethods, awkward.JaggedArray)
