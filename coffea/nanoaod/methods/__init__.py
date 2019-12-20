from .common import METVector, LorentzVector, Candidate
from .leptons import Electron, Muon, Photon, Tau
from .jets import Jet, FatJet
from .generator import GenParticle, GenVisTau


collection_methods = {
    'CaloMET': METVector,
    'ChsMET': METVector,
    'GenMET': METVector,
    'MET': METVector,
    'METFixEE2017': METVector,
    'PuppiMET': METVector,
    'RawMET': METVector,
    'TkMET': METVector,
    # pseudo-lorentz: pt, eta, phi, mass=0
    'IsoTrack': LorentzVector,
    'SoftActivityJet': LorentzVector,
    'TrigObj': LorentzVector,
    # True lorentz: pt, eta, phi, mass
    'FatJet': FatJet,
    'GenDressedLepton': LorentzVector,
    'GenJet': LorentzVector,
    'GenJetAK8': FatJet,
    'Jet': Jet,
    'LHEPart': LorentzVector,
    'SV': LorentzVector,
    'SubGenJetAK8': LorentzVector,
    'SubJet': LorentzVector,
    # Candidate: LorentzVector + charge
    'Electron': Electron,
    'Muon': Muon,
    'Photon': Photon,
    'Tau': Tau,
    'GenVisTau': GenVisTau,
    # special
    'GenPart': GenParticle,
}

__all__ = [
    'METVector',
    'LorentzVector',
    'Candidate',
    'Electron',
    'Muon',
    'Photon',
    'Tau',
    'Jet',
    'FatJet',
    'GenParticle',
    'GenVisTau',
    'collection_methods',
]
