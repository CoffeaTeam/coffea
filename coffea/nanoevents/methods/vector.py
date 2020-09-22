"""2D, 3D, and Lorentz vector class mixins

These mixins will eventually be superceded by the `vector <https://github.com/scikit-hep/vector>`__ library,
which will hopefully be feature-compatible.

A small example::

    import numpy as np
    import awkward1 as ak
    from coffea.nanoevents.methods import vector
    ak.behavior.update(vector.behavior)

    n = 1000

    vec = ak.zip(
        {
            "x": np.random.normal(size=n),
            "y": np.random.normal(size=n),
            "z": np.random.normal(size=n),
        },
        with_name="ThreeVector",
    )

    vec4 = ak.zip(
        {
            "pt": vec.r,
            "eta": -np.log(np.tan(vec.theta/2)),
            "phi": vec.phi,
            "mass": np.full(n, 1.),
        },
        with_name="PtEtaPhiMLorentzVector",
    )

    assert np.allclose(np.array(vec4.x), np.array(vec.x))
    assert np.allclose(np.array(vec4.y), np.array(vec.y))
    assert np.allclose(np.array(vec4.z), np.array(vec.z))
    assert np.allclose(np.array(abs(2*vec + vec4) / abs(vec)), 3)

"""
import numbers
import numpy
import awkward1


behavior = {}


@awkward1.mixin_class(behavior)
class TwoVector:
    """A cartesian 2-dimensional vector

    A heavy emphasis towards a momentum vector interpretation is assumed.
    Properties this class requires: x, y
    """

    @property
    def r(self):
        return numpy.sqrt(self.r2)

    @property
    def phi(self):
        return numpy.arctan2(self.y, self.x)

    @property
    def px(self):
        return self.x

    @property
    def py(self):
        return self.y

    @property
    def r2(self):
        return self.x ** 2 + self.y ** 2

    @property
    def pt2(self):
        return self.r2

    @property
    def pt(self):
        return self.r

    @awkward1.mixin_class_method(numpy.absolute)
    def abs(self):
        return self.r

    @awkward1.mixin_class_method(numpy.add, {"TwoVector"})
    def add(self, other):
        return awkward1.zip(
            {"x": self.x + other.x, "y": self.y + other.y},
            with_name="TwoVector",
        )

    def sum(self, axis=-1):
        out = awkward1.zip(
            {
                "x": awkward1.sum(self.x, axis=axis),
                "y": awkward1.sum(self.y, axis=axis),
            },
            with_name="TwoVector",
            highlevel=False,
        )
        return awkward1._util.wrap(out, cache=self.cache, behavior=self.behavior)

    @awkward1.mixin_class_method(numpy.multiply, {numbers.Number})
    def prod(self, other):
        return awkward1.zip(
            {"x": self.x * other, "y": self.y * other},
            with_name="TwoVector",
        )

    def delta_phi(self, other):
        """Compute difference in angle

        Returns a value within [-pi, pi)
        """
        return (self.phi - other.phi + numpy.pi) % (2 * numpy.pi) - numpy.pi


@awkward1.mixin_class(behavior)
class PolarTwoVector(TwoVector):
    """A polar coordinate 2-d vector

    This class overloads the properties r, phi with getitem accessors
    and provides properties x, y
    Some additional properties are overridden for performance
    """

    @property
    def x(self):
        return self.r * numpy.cos(self.phi)

    @property
    def y(self):
        return self.r * numpy.sin(self.phi)

    @property
    def r(self):
        return self["r"]

    @property
    def phi(self):
        return self["phi"]

    @property
    def r2(self):
        return self.r ** 2


@awkward1.mixin_class(behavior)
class ThreeVector(TwoVector):
    """A cartesian 3-dimensional vector

    A heavy emphasis towards a momentum vector interpretation is assumed.
    Properties this class requires: x, y, z
    """

    @property
    def pz(self):
        return self.z

    @property
    def rho2(self):
        return self.r2 + self.z ** 2

    @property
    def rho(self):
        return numpy.sqrt(self.rho2)

    @property
    def theta(self):
        return numpy.arctan2(self.r, self.z)

    @property
    def p2(self):
        return self.rho2

    @property
    def p(self):
        return self.rho

    @awkward1.mixin_class_method(numpy.absolute)
    def abs(self):
        return self.p

    @awkward1.mixin_class_method(numpy.add, {"ThreeVector"})
    def add(self, other):
        return awkward1.zip(
            {"x": self.x + other.x, "y": self.y + other.y, "z": self.z + other.z},
            with_name="ThreeVector",
        )

    def sum(self, axis=-1):
        out = awkward1.zip(
            {
                "x": awkward1.sum(self.x, axis=axis),
                "y": awkward1.sum(self.y, axis=axis),
                "z": awkward1.sum(self.z, axis=axis),
            },
            with_name="ThreeVector",
            highlevel=False,
        )
        return awkward1._util.wrap(out, cache=self.cache, behavior=self.behavior)

    @awkward1.mixin_class_method(numpy.multiply, {numbers.Number})
    def prod(self, other):
        return awkward1.zip(
            {"x": self.x * other, "y": self.y * other, "z": self.z * other},
            with_name="ThreeVector",
        )


@awkward1.mixin_class(behavior)
class SphericalThreeVector(ThreeVector, PolarTwoVector):
    """A spherical coordinate 3-d vector

    This class overloads the properties rho, theta with getitem accessors
    and provides properties r, z
    Some additional properties are overridden for performance
    """

    @property
    def r(self):
        return self.rho * numpy.sin(self.theta)

    @property
    def z(self):
        return self.rho * numpy.cos(self.theta)

    @property
    def rho(self):
        return self["rho"]

    @property
    def theta(self):
        return self["theta"]

    @property
    def p(self):
        return self.rho

    @property
    def p2(self):
        return self.rho ** 2


@awkward1.mixin_class(behavior)
class LorentzVector(ThreeVector):
    """A cartesian Lorentz vector

    A heavy emphasis towards a momentum vector interpretation is assumed.
    + - - - metric
    Properties this class requires: x, y, z, t
    """

    @property
    def energy(self):
        return self.t

    @property
    def eta(self):
        return numpy.arcsinh(self.z / self.r)

    @property
    def mass2(self):
        return self.t ** 2 - self.p2

    @property
    def mass(self):
        return numpy.sqrt(self.mass2)

    @awkward1.mixin_class_method(numpy.absolute)
    def abs(self):
        return self.mass

    @awkward1.mixin_class_method(numpy.add, {"LorentzVector"})
    def add(self, other):
        return awkward1.zip(
            {
                "x": self.x + other.x,
                "y": self.y + other.y,
                "z": self.z + other.z,
                "t": self.t + other.t,
            },
            with_name="LorentzVector",
        )

    def sum(self, axis=-1):
        out = awkward1.zip(
            {
                "x": awkward1.sum(self.x, axis=axis),
                "y": awkward1.sum(self.y, axis=axis),
                "z": awkward1.sum(self.z, axis=axis),
                "t": awkward1.sum(self.t, axis=axis),
            },
            with_name="LorentzVector",
            highlevel=False,
        )
        return awkward1._util.wrap(out, cache=self.cache, behavior=self.behavior)

    @awkward1.mixin_class_method(numpy.multiply, {numbers.Number})
    def prod(self, other):
        return awkward1.zip(
            {
                "x": self.x * other,
                "y": self.y * other,
                "z": self.z * other,
                "t": self.t * other,
            },
            with_name="LorentzVector",
        )

    def delta_r2(self, other):
        return (self.eta - other.eta) ** 2 + self.delta_phi(other) ** 2

    def delta_r(self, other):
        return numpy.sqrt(self.delta_r2(other))

    def nearest(
        self,
        other,
        axis=1,
        metric=lambda a, b: a.delta_r(b),
        return_metric=False,
        threshold=None,
    ):
        """Return nearest object to this one

        Finds item in ``other`` satisfying ``min(metric(self, other))``.
        The two arrays should be broadcast-compatible on all axes other than the specified
        axis, which will be used to form a cartesian product. If axis=None, broadcast arrays directly.
        The return shape will be that of ``self``.

        Parameters
        ----------
            other : awkward1.Array
                Another array with same shape in all but ``axis``
            axis : int, optional
                The axis to form the cartesian product (default 1). If None, the metric
                is directly evaluated on the input arrays (i.e. they should broadcast)
            metric : callable
                A function of two arguments, returning a scalar. The default metric is `delta_r`.
            return_metric : bool, optional
                If true, return both the closest object and its metric (default false)
            threshold : Number, optional
                If set, any objects with ``metric > threshold`` will be masked from the result
        """
        if axis is None:
            a, b = self, other
            # NotImplementedError: ak.firsts with axis=-1
            axis = other.layout.purelist_depth - 2
        else:
            a, b = awkward1.unzip(
                awkward1.cartesian([self, other], axis=axis, nested=True)
            )
        mval = metric(a, b)
        # prefer keepdims=True: awkward-1.0 #434
        mmin = awkward1.singletons(awkward1.argmin(mval, axis=axis + 1))
        out = awkward1.firsts(b[mmin], axis=axis + 1)
        metric = awkward1.firsts(mval[mmin], axis=axis + 1)
        if threshold is not None:
            out = out.mask[metric <= threshold]
        if return_metric:
            return out, metric
        return out


@awkward1.mixin_class(behavior)
class PtEtaPhiMLorentzVector(LorentzVector, SphericalThreeVector):
    """A Lorentz vector using pseudorapidity and mass

    This class overloads the pt, eta, phi, mass properties with getitem accessors
    and provides properties rho, theta, r, z, t
    Some additional properties are overridden for performance
    """

    @property
    def pt(self):
        return self["pt"]

    @property
    def eta(self):
        return self["eta"]

    @property
    def phi(self):
        return self["phi"]

    @property
    def mass(self):
        return self["mass"]

    @property
    def rho(self):
        return self.pt * numpy.cosh(self.eta)

    @property
    def theta(self):
        raise NotImplementedError

    @property
    def r(self):
        return self.pt

    @property
    def z(self):
        return self.pt * numpy.sinh(self.eta)

    @property
    def t(self):
        return numpy.hypot(self.rho, self.mass)

    @property
    def rho2(self):
        return self.rho ** 2

    @property
    def mass2(self):
        return self.mass ** 2

    @awkward1.mixin_class_method(numpy.multiply, {numbers.Number})
    def prod(self, other):
        return awkward1.zip(
            {
                "pt": self.pt * other,
                "eta": self.eta,
                "phi": self.phi,
                "mass": self.mass * other,
            },
            with_name="PtEtaPhiMLorentzVector",
        )


@awkward1.mixin_class(behavior)
class PtEtaPhiELorentzVector(LorentzVector, SphericalThreeVector):
    """A Lorentz vector using pseudorapidity and energy

    This class overloads the pt, eta, phi, energy, t properties with getitem accessors
    and provides properties rho, theta, r, z
    Some additional properties are overridden for performance
    """

    @property
    def pt(self):
        return self["pt"]

    @property
    def eta(self):
        return self["eta"]

    @property
    def phi(self):
        return self["phi"]

    @property
    def energy(self):
        return self["energy"]

    @property
    def t(self):
        return self["energy"]

    @property
    def rho(self):
        return self.pt * numpy.cosh(self.eta)

    @property
    def theta(self):
        raise NotImplementedError

    @property
    def r(self):
        return self.pt

    @property
    def z(self):
        return self.pt * numpy.sinh(self.eta)

    @property
    def rho2(self):
        return self.rho ** 2

    @awkward1.mixin_class_method(numpy.multiply, {numbers.Number})
    def prod(self, other):
        return awkward1.zip(
            {
                "pt": self.pt * other,
                "eta": self.eta,
                "phi": self.phi,
                "energy": self.energy * other,
            },
            with_name="PtEtaPhiELorentzVector",
        )
