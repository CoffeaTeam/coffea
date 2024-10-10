import awkward
import dask_awkward
import numba
import numpy
from dask_awkward.lib.core import dask_property

from coffea.nanoevents import transforms
from coffea.nanoevents.methods import base, vector

behavior = {}
behavior.update(base.behavior)


class _FCCEvents(behavior["NanoEvents"]):
    def __repr__(self):
        return "FCC Events"


behavior["NanoEvents"] = _FCCEvents


def _set_repr_name(classname):
    def namefcn(self):
        return classname

    behavior[classname].__repr__ = namefcn


def map_index_to_array(array, index, axis=1):
    """
    DESCRIPTION: Creates a slice of input array according to the input index.
    INPUTS: array (Singly nested)
            index (Singly or Doubly nested)
            axis (By default 1, use axis = 2 if index is doubly nested )
    EXAMPLE:
            a = awkward.Array([
                [44,33,23,22],
                [932,24,456,78],
                [22,345,78,90,98,24]
            ])

            a_index = awkward.Array([
                [0,1,2],
                [0,1],
                []
            ])

            a2_index = awkward.Array([
                [[0],[0,1],[2]],
                [[0,1]],
                []
            ])
            >> map_index_to_array(a, a_index)
                [[44, 33, 23],
                 [932, 24],
                 []]
                ---------------------
                type: 3 * var * int64
            >> map_index_to_array(a, a2_index, axis=2)
                [[[44], [44, 33], [23]],
                 [[932, 24]],
                 []]
                ---------------------------
                type: 3 * var * var * int64

    """
    if axis == 1:
        return array[index]
    elif axis == 2:
        axis2_counts_array = awkward.num(index, axis=axis)
        flat_axis2_counts_array = awkward.flatten(axis2_counts_array, axis=1)
        flat_index = awkward.flatten(index, axis=axis)
        trimmed_flat_array = array[flat_index]
        trimmed_array = awkward.unflatten(
            trimmed_flat_array, flat_axis2_counts_array, axis=1
        )
        return trimmed_array
    else:
        raise AttributeError("Only axis = 1 or axis = 2 supported at the moment.")


# Function required to create a range array from a begin and end array
@numba.njit
def index_range_numba_wrap(begin_end, builder):
    for ev in begin_end:
        builder.begin_list()
        for j in ev:
            builder.begin_list()
            for k in range(j[0], j[1]):
                builder.integer(k)
            builder.end_list()
        builder.end_list()
    return builder


def index_range(begin, end):
    """
    Function required to create a range array from a begin and end array
    Example: If,
            begin = [
                        [0, 2, 4, 3, ...],
                        [1, 0, 4, 6, ...]
                        ...
                    ]
            end = [
                        [1, 2, 5, 5, ...],
                        [3, 1, 7, 6, ...]
                        ...
                    ]
            then, output is,
            output = [
                        [[0], [], [4], [3,4], ...],
                        [[1,2], [0], [4,5,6], [], ...]
                        ...
                    ]
    """
    begin_end = awkward.concatenate(
        (begin[:, :, numpy.newaxis], end[:, :, numpy.newaxis]), axis=2
    )
    if awkward.backend(begin) == "typetracer" or awkward.backend(end) == "typetracer":
        # To make the function dask compatible
        # here we fake the output of numba wrapper function since
        # operating on length-zero data returns the wrong layout!
        # We need the axis 2, therefore, we should return the typetracer layout of [[[]]]
        awkward.typetracer.length_zero_if_typetracer(
            begin
        )  # force touching of the necessary data
        awkward.typetracer.length_zero_if_typetracer(
            end
        )  # force touching of the necessary data
        return awkward.Array(
            awkward.Array([[[0]]]).layout.to_typetracer(forget_length=True)
        )

    return index_range_numba_wrap(begin_end, awkward.ArrayBuilder()).snapshot()


@awkward.mixin_class(behavior)
class MomentumCandidate(vector.LorentzVector):
    """A Lorentz vector with charge

    This mixin class requires the parent class to provide items `px`, `py`, `pz`, `E`, and `charge`.
    """

    @awkward.mixin_class_method(numpy.add, {"MomentumCandidate"})
    def add(self, other):
        """Add two candidates together elementwise using `px`, `py`, `pz`, `E`, and `charge` components"""
        return awkward.zip(
            {
                "px": self.px + other.px,
                "py": self.py + other.py,
                "pz": self.pz + other.pz,
                "E": self.E + other.E,
                "charge": self.charge + other.charge,
            },
            with_name="MomentumCandidate",
            behavior=self.behavior,
        )

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `px`, `py`, `pz`, `E`, and `charge` components"""
        return awkward.zip(
            {
                "px": awkward.sum(self.px, axis=axis),
                "py": awkward.sum(self.py, axis=axis),
                "pz": awkward.sum(self.pz, axis=axis),
                "E": awkward.sum(self.E, axis=axis),
                "charge": awkward.sum(self.charge, axis=axis),
            },
            with_name="MomentumCandidate",
            behavior=self.behavior,
        )

    @property
    def absolute_mass(self):
        return numpy.sqrt(numpy.abs(self.mass2))


behavior.update(
    awkward._util.copy_behaviors(vector.LorentzVector, MomentumCandidate, behavior)
)

MomentumCandidateArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MomentumCandidateArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MomentumCandidateArray.ProjectionClass4D = vector.LorentzVectorArray  # noqa: F821
MomentumCandidateArray.MomentumClass = MomentumCandidateArray  # noqa: F821


@awkward.mixin_class(behavior)
class MCParticle(MomentumCandidate, base.NanoCollection):
    """Generated Monte Carlo particles"""
    
    @property
    def alt_get_daughters_index(self):
        """
        Obtain the indexes of the daughters of each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx1 collection
        - The Particleidx1.index contains info about the daughters
        """
        ranges = index_range(self.daughters.begin, self.daughters.end)
        
        return awkward.values_astype(
            map_index_to_array(self._events().Particleidx1.index, ranges, axis=2),
            "int64",
        )
    
    
    # Daughters
    @dask_property
    def get_daughters_index(self):
        """
        Obtain the indexes of the daughters of each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx1 collection
        - The Particleidx1.index contains info about the daughters
        """
        ranges = index_range(self.daughters.begin, self.daughters.end)
        return awkward.values_astype(
            map_index_to_array(self._events().Particleidx1.index, ranges, axis=2),
            "int64",
        )

    @dask_property
    def get_daughters(self):
        """
        Obtain the actual MCParticle daughters to each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx1 collection
        - The Particleidx1.index contains info about the daughters
        """
        return map_index_to_array(self, self.get_daughters_index, axis=2)

    @get_daughters.dask
    def get_daughters(self, dask_array):
        """
        Obtain the actual MCParticle daughters to each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx1 collection
        - The Particleidx1.index contains info about the daughters
        """
        return map_index_to_array(dask_array, dask_array.get_daughters_index, axis=2)

    # Parents
    @dask_property
    def get_parents_index(self):
        """
        Obtain the indexes of the parents of each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx0 collection
        - The Particleidx0.index contains info about the parents
        """
        ranges = index_range(self.parents.begin, self.parents.end)
        # rangesG = index_range(self.parents.beginG, self.parents.endG)
        # Explore how to map the global index to produces doubly nested output
        return awkward.values_astype(
            map_index_to_array(self._events().Particleidx0.index, ranges, axis=2),
            "int64",
        )

    @get_parents_index.dask
    def get_parents_index(self, dask_array):
        """
        Obtain the indexes of the parents of each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx0 collection
        - The Particleidx0.index contains info about the parents

        Note: Seems like all the functions need to mapped manually
        """
        ranges = dask_awkward.map_partitions(
            index_range, dask_array.parents.begin, dask_array.parents.end
        )
        daughters = dask_awkward.map_partitions(
            map_index_to_array, dask_array._events().Particleidx0.index, ranges, axis=2
        )
        return awkward.values_astype(daughters, "int32")

    @dask_property
    def get_parents(self):
        """
        Obtain the actual MCParticle parents to each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx0 collection
        - The Particleidx0.index contains info about the parents
        """
        return map_index_to_array(self, self.get_parents_index, axis=2)

    @get_parents.dask
    def get_parents(self, dask_array):
        """
        Obtain the actual MCParticle parents to each and every MCParticle
        - The output is a doubly nested awkward array
        - Needs the presence of Particleidx0 collection
        - The Particleidx0.index contains info about the parents
        """
        return map_index_to_array(dask_array, dask_array.get_parents_index, axis=2)


_set_repr_name("MCParticle")
behavior.update(awkward._util.copy_behaviors(MomentumCandidate, MCParticle, behavior))

MCParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
MCParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
MCParticleArray.ProjectionClass4D = MCParticleArray  # noqa: F821
MCParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class ReconstructedParticle(MomentumCandidate, base.NanoCollection):
    """Reconstructed particle"""

    def match_collection(self, idx):
        """Returns matched particles"""
        return self[idx.index]
    
    @dask_property
    def match_muons(self):
        """Returns matched muons"""
        m = self._events().ReconstructedParticles._apply_global_index(self.Muonidx0_indexGlobal)
        return awkward.drop_none(m, behavior = self.behavior)
        
    @match_muons.dask
    def match_muons(self, dask_array):
        """Returns matched muons"""
        m = dask_array._events().ReconstructedParticles._apply_global_index(dask_array.Muonidx0_indexGlobal)
        return awkward.drop_none(m, behavior = self.behavior)
        
    @dask_property
    def match_electrons(self):
        """Returns matched electrons"""
        e = self._events().ReconstructedParticles._apply_global_index(self.Electronidx0_indexGlobal)
        return awkward.drop_none(e, behavior = self.behavior)
        
    @match_electrons.dask
    def match_electrons(self, dask_array):
        """Returns matched electrons"""
        e = dask_array._events().ReconstructedParticles._apply_global_index(dask_array.Electronidx0_indexGlobal)
        return awkward.drop_none(e, behavior = self.behavior)

    @dask_property
    def match_gen(self):
        # The indices may have a size greater or smaller than the Particle collection, in axis=1
        prepared = self._events().Particle[self._events().MCRecoAssociations.mc.index]
        return prepared._apply_global_index(self.MCRecoAssociationsidx0_indexGlobal)

    @match_gen.dask
    def match_gen(self, dask_array):
        # The indices may have a size greater or smaller than the Particle collection, in axis=1
        prepared = dask_array._events().Particle[dask_array._events().MCRecoAssociations.mc.index]
        return prepared._apply_global_index(dask_array.MCRecoAssociationsidx0_indexGlobal)
    
_set_repr_name("ReconstructedParticle")
behavior.update(
    awkward._util.copy_behaviors(MomentumCandidate, ReconstructedParticle, behavior)
)

ReconstructedParticleArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ReconstructedParticleArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ReconstructedParticleArray.ProjectionClass4D = ReconstructedParticleArray  # noqa: F821
ReconstructedParticleArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class RecoMCParticleLink(base.NanoCollection):
    """MCRecoParticleAssociation objects."""

    @property
    def reco_mc_index(self):
        """
        Returns an array of indices mapping to generator particles for each reconstructed particle
        """
        arr_reco = self.reco.index[:, :, numpy.newaxis]
        arr_mc = self.mc.index[:, :, numpy.newaxis]

        joined_indices = awkward.concatenate((arr_reco, arr_mc), axis=2)

        return joined_indices

    @dask_property
    def reco_mc(self):
        """
        Returns an array of records mapping to generator particle record for each reconstructed particle record
        - Needs 'ReconstructedParticles' and 'Particle' collections
        """
        reco_index = self.reco.index
        mc_index = self.mc.index
        r = self._events().ReconstructedParticles[reco_index][:, :, numpy.newaxis]
        m = self._events().Particle[mc_index][:, :, numpy.newaxis]

        return awkward.concatenate((r, m), axis=2)

    @reco_mc.dask
    def reco_mc(self, dask_array):
        """
        Returns an array of records mapping to generator particle record for each reconstructed particle record
        - Needs 'ReconstructedParticles' and 'Particle' collections
        """
        reco_index = dask_array.reco.index
        mc_index = dask_array.mc.index
        r = dask_array._events().ReconstructedParticles[reco_index][:, :, numpy.newaxis]
        m = dask_array._events().Particle[mc_index][:, :, numpy.newaxis]

        return awkward.concatenate((r, m), axis=2)


_set_repr_name("RecoMCParticleLink")

RecoMCParticleLinkArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
RecoMCParticleLinkArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
RecoMCParticleLinkArray.ProjectionClass4D = RecoMCParticleLinkArray  # noqa: F821
RecoMCParticleLinkArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class ParticleID(base.NanoCollection):
    """ParticleID collection"""


_set_repr_name("ParticleID")

ParticleIDArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ParticleIDArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ParticleIDArray.ProjectionClass4D = ParticleIDArray  # noqa: F821
ParticleIDArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class ObjectID(base.NanoCollection):
    """
    Generic Object ID storage, pointing to another collection
    - All the idx collections are assigned this mixin

    Note: The Hash tagged '#' branches have the <podio::ObjectID> type
    """


_set_repr_name("ObjectID")

ObjectIDArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ObjectIDArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ObjectIDArray.ProjectionClass4D = ObjectIDArray  # noqa: F821
ObjectIDArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class Cluster(base.NanoCollection):
    """
    Clusters

    Note: I could not find much info on this, to build its methods
    """


_set_repr_name("Cluster")

ClusterArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
ClusterArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
ClusterArray.ProjectionClass4D = ClusterArray  # noqa: F821
ClusterArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821


@awkward.mixin_class(behavior)
class Track(base.NanoCollection):
    """
    Tracks

    Note: I could not find much info on this, to build its methods
    """


_set_repr_name("Track")

TrackArray.ProjectionClass2D = vector.TwoVectorArray  # noqa: F821
TrackArray.ProjectionClass3D = vector.ThreeVectorArray  # noqa: F821
TrackArray.ProjectionClass4D = TrackArray  # noqa: F821
TrackArray.MomentumClass = vector.LorentzVectorArray  # noqa: F821
