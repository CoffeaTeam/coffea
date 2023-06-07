import awkward

from coffea.nanoevents.methods import base, vector

behavior = {}


@awkward.mixin_class(behavior)
class MCTruthParticle(vector.LorentzVectorM):
    """Generated Monte Carlo particles."""


@awkward.mixin_class(behavior)
class RecoParticle(vector.LorentzVector, base.NanoCollection):
    """Reconstructed particles."""

    @property
    def matched_gen(self):
        """Returns an array of matched generator particle objects for each reconstructed particle."""

        matched_particles = self.behavior[
            "__original_array__"
        ]().MCParticlesSkimmed._apply_global_index(
            self.behavior["__original_array__"]().RecoMCTruthLink.reco_mc_index,
        )

        return matched_particles


@awkward.mixin_class(behavior)
class Cluster(vector.PtEtaPhiELorentzVector):
    """Clusters. Will be updated to have linking like RecoParticles."""


@awkward.mixin_class(behavior)
class Track:
    """Tracks. Will be updated to have linking like RecoParticles."""


@awkward.mixin_class(behavior)
class ParticleLink:
    """MCRecoParticleAssociation objects."""

    @property
    def reco_mc_index(self):
        """
        returns an array of indices mapping to generator particles for each reconstructed particle
        """
        arr_reco = self.reco_index
        arr_mc = self.mc_index

        # this is just to shape the index array properly
        sorted_reco = arr_reco[awkward.argsort(arr_reco)]
        sorted_mc = arr_mc[awkward.argsort(arr_reco)]
        proper_indices = awkward.unflatten(
            sorted_mc, awkward.flatten(awkward.run_lengths(sorted_reco), axis=1), axis=1
        )

        return proper_indices
