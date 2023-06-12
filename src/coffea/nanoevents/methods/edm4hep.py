import awkward
import numpy

from coffea.nanoevents.methods import base, vector

PION_MASS = 0.13957018 # GeV

behavior = {}


@awkward.mixin_class(behavior)
class MCTruthParticle(vector.LorentzVectorM):
    """ Generated Monte Carlo particles. 
    """


@awkward.mixin_class(behavior)
class RecoParticle(vector.LorentzVector, base.NanoCollection):
    """ Reconstructed particles.
    """

    @property
    def matched_gen(self):
        """ Returns an array of matched generator particle objects for each reconstructed particle.
        """

        matched_particles = self.behavior[
            "__original_array__"
        ]().MCParticlesSkimmed._apply_global_index(
            self.behavior["__original_array__"]().RecoMCTruthLink.reco_mc_index,
        )

        return matched_particles


@awkward.mixin_class(behavior)
class Cluster(vector.PtThetaPhiELorentzVector):
    """ Clusters. Will be updated to have linking like RecoParticles.
    """


@awkward.mixin_class(behavior)
class Track(vector.LorentzVectorM,base.NanoEvents):
    """ Tracks. Will be updated to have linking like RecoParticles.
    """
    @property
    def pt(self):
        r"""transverse momentum
        mag :: magnetic field strength in T

        source: https://github.com/PandoraPFA/MarlinPandora/blob/master/src/TrackCreator.cc#LL521 
        """
        metadata = self.behavior[
            "__original_array__"
        ]().get_metadata()
        
        if metadata == None or 'b_field' not in metadata.keys(): 
            print('Track momentum requires value of magnetic field. \n Please have \'metadata\' argument in from_root function have key \'b_field\' with the value of magnetic field.')
            raise ValueError('Track momentum requires value of magnetic field. \n Please have \'metadata\' argument in from_root function have key \'b_field\' with the value of magnetic field.')
        else:
            b_field = metadata['b_field']
            return b_field*2.99792e-4/numpy.abs(self["omega"]) 

    @property
    def phi(self):
        r"""x momentum
        """
        return self["phi"]

    @property
    def x(self):
        r"""x momentum
        """
        return numpy.cos(self["phi"])*self.pt
    
    @property
    def y(self):
        r"""y momentum
        """
        return numpy.sin(self["phi"])*self.pt
    
    @property
    def z(self):
        r"""z momentum
        """
        return self["tanLambda"]*self.pt
    

    @property
    def mass(self):
        r""" mass of the track - assumed to be the mass of a pion 
        source: https://github.com/iLCSoft/MarlinTrk/blob/c53d868979ef6db26077746ce264633819ffcf4f/src/MarlinAidaTTTrack.cc#LL54C3-L58C3
        """
        return PION_MASS*awkward.ones_like(self["omega"])
    


@awkward.mixin_class(behavior)
class ParticleLink:
    """ MCRecoParticleAssociation objects.
    """

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
