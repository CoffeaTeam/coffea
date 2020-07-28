"""Awkward 0.x NanoAOD collection methods

This package provides the object method mixins for the various
types of NanoAOD collections, as well as a default mapping for collection
names to methods.
"""
from .common import METVector, LorentzVector, Candidate
from .leptons import Electron, Muon, Photon, FsrPhoton, Tau
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
    'GenJetAK8': LorentzVector,
    'Jet': Jet,
    'LHEPart': LorentzVector,
    'SV': LorentzVector,
    'SubGenJetAK8': LorentzVector,
    'SubJet': LorentzVector,
    'PFCands': LorentzVector,  # available in NanoAODJMAR
    # Candidate: LorentzVector + charge
    'Electron': Electron,
    'Muon': Muon,
    'Photon': Photon,
    'Tau': Tau,
    'GenVisTau': GenVisTau,
    'FsrPhoton': FsrPhoton,
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
    'FsrPhoton',
    'Tau',
    'Jet',
    'FatJet',
    'GenParticle',
    'GenVisTau',
    'collection_methods',
]
