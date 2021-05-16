import awkward
import numpy
from coffea.nanoevents.methods import base, vector


behavior = {}
behavior.update(base.behavior)
behavior.update(vector.behavior)


def _element_link(target_collection, global_index, key):
    global_index = awkward.where(key != 0, global_index, -1)
    return target_collection._apply_global_index(global_index)


@awkward.mixin_class(behavior)
class xAODParticle(vector.PtEtaPhiMLorentzVector, base.NanoCollection):
    @property
    def mass(self):
        return self.m


@awkward.mixin_class(behavior)
class xAODTrackParticle(vector.LorentzVector, base.NanoCollection):
    "see https://gitlab.cern.ch/atlas/athena/-/blob/21.2/Event/xAOD/xAODTracking/Root/TrackParticle_v1.cxx#L82"

    @property
    def theta(self):
        return self["theta"]

    @property
    def phi(self):
        return self["phi"]

    @property
    def p(self):
        return 1.0 / numpy.abs(self.qOverP)

    @property
    def x(self):
        return self.p * numpy.sin(self.theta) * numpy.cos(self.phi)

    @property
    def y(self):
        return self.p * numpy.sin(self.theta) * numpy.sin(self.phi)

    @property
    def z(self):
        return self.p * numpy.cos(self.theta)

    @property
    def t(self):
        return numpy.sqrt(139.570 ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)


@awkward.mixin_class(behavior)
class xAODMuon(xAODParticle):
    @property
    def trackParticle(self):
        return _element_link(
            self._events().CombinedMuonTrackParticles,
            self[
                "combinedTrackParticleLink.m_persIndex__G__CombinedMuonTrackParticles"
            ],
            self["combinedTrackParticleLink.m_persKey"],
        )


@awkward.mixin_class(behavior)
class xAODElectron(xAODParticle):
    @property
    def trackParticles(self):
        return _element_link(
            self._events().GSFTrackParticles,
            self.trackParticleLinks__G__GSFTrackParticles,
            self.trackParticleLinks.m_persKey,
        )

    @property
    def trackParticle(self):
        trackParticles = self.trackParticles
        return self.trackParticles[
            tuple([slice(None) for i in range(trackParticles.ndim - 1)] + [0])
        ]
