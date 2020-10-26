import uproot_methods
from coffea.util import awkward


class JaggedTLorentzVectorArray(uproot_methods.classes.TLorentzVector.ArrayMethods, awkward.JaggedArray):
    """TLorentzVector methods mixed in with an awkward0 JaggedArray"""

    pass
