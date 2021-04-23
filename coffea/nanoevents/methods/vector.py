"""2D, 3D, and Lorentz vector class mixins

These mixins will eventually be superceded by the `vector <https://github.com/scikit-hep/vector>`__ library,
which will hopefully be feature-compatible. The 2D vector provides cartesian and polar coordinate attributes,
where ``r`` represents the polar distance from the origin..  The 3D vector provides cartesian and spherical coordinates,
where ``rho`` represents the 3D distance from the origin and ``r`` is the axial distance from the z axis, so that it can
subclass the 2D vector. The Lorentz vector also subclasses the 3D vector, adding ``t`` as the fourth
cartesian coordinate. Aliases typical of momentum vectors are also provided.

A small example::

    import numpy as np
    import awkward as ak
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
import awkward


behavior = {}


@awkward.mixin_class(behavior)
class TwoVector:
    """A cartesian 2-dimensional vector

    A heavy emphasis towards a momentum vector interpretation is assumed, hence
    properties like `px` and `py` are provided in addition to `x` and `y`.

    This mixin class requires the parent class to provide items `x` and `y`.
    """

    @property
    def r(self):
        r"""Distance from origin in XY plane

        :math:`\sqrt{x^2+y^2}`
        """
        return numpy.sqrt(self.r2)

    @property
    def phi(self):
        r"""Polar angle relative to X axis

        :math:`\text{arctan2}(y, x)`
        """
        return numpy.arctan2(self.y, self.x)

    @property
    def px(self):
        """Alias for `x`"""
        return self.x

    @property
    def py(self):
        """Alias for `y`"""
        return self.y

    @property
    def r2(self):
        """Squared `r`"""
        return self.x ** 2 + self.y ** 2

    @property
    def pt2(self):
        """Alias for `r2`"""
        return self.r2

    @property
    def pt(self):
        """Alias for `r`"""
        return self.r

    @awkward.mixin_class_method(numpy.absolute)
    def absolute(self):
        """Returns magnitude of the 2D vector

        Alias for `r`
        """
        return self.r

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {"x": -self.x, "y": -self.y},
            with_name="TwoVector",
        )

    @awkward.mixin_class_method(numpy.add, {"TwoVector"})
    def add(self, other):
        """Add two vectors together elementwise using `x` and `y` components"""
        return awkward.zip(
            {"x": self.x + other.x, "y": self.y + other.y},
            with_name="TwoVector",
        )

    @awkward.mixin_class_method(numpy.subtract, {"TwoVector"})
    def subtract(self, other):
        """Substract a vector from another elementwise using `x` and `y` compontents"""
        return awkward.zip(
            {"x": self.x - other.x, "y": self.y - other.y},
            with_name="TwoVector",
        )

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `x` and `y` components"""
        out = awkward.zip(
            {
                "x": awkward.sum(self.x, axis=axis),
                "y": awkward.sum(self.y, axis=axis),
            },
            with_name="TwoVector",
            highlevel=False,
        )
        return awkward.Array(out, behavior=self.behavior)

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x` and `y` components"""
        return awkward.zip(
            {"x": self.x * other, "y": self.y * other},
            with_name="TwoVector",
        )

    @awkward.mixin_class_method(numpy.divide, {numbers.Number})
    def divide(self, other):
        """Divide this vector by a scalar elementwise using its cartesian components

        This is realized by using the multiplication functionality"""
        return self.multiply(1 / other)

    def delta_phi(self, other):
        """Compute difference in angle between two vectors

        Returns a value within [-pi, pi)
        """
        return (self.phi - other.phi + numpy.pi) % (2 * numpy.pi) - numpy.pi

    def dot(self, other):
        """Compute the dot product of two vectors"""
        return self.x * other.x + self.y * other.y

    @property
    def unit(self):
        """Unit vector, a vector of length 1 pointing in the same direction"""
        return self / self.r


@awkward.mixin_class(behavior)
class PolarTwoVector(TwoVector):
    """A polar coordinate 2-dimensional vector

    This mixin class requires the parent class to provide items `r` and `phi`.
    Some additional properties are overridden for performance
    """

    @property
    def x(self):
        r"""Cartesian x value

        :math:`r \cos{\phi}`
        """
        return self.r * numpy.cos(self.phi)

    @property
    def y(self):
        r"""Cartesian y value

        :math:`r \sin{\phi}`
        """
        return self.r * numpy.sin(self.phi)

    @property
    def r(self):
        r"""Distance from origin in XY plane

        :math:`\sqrt{x^2+y^2}`
        """
        return self["r"]

    @property
    def phi(self):
        r"""Azimuthal angle relative to X axis in XY plane

        :math:`\text{arctan2}(y, x)`
        """
        return self["phi"]

    @property
    def r2(self):
        """Squared `r`"""
        return self.r ** 2

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using using `x` and `y` components

        In reality, this directly adjusts `r` and `phi` for performance
        """
        return awkward.zip(
            {
                "r": self.r * abs(other),
                "phi": self.phi % (2 * numpy.pi) - (numpy.pi * (other < 0)),
            },
            with_name="PolarTwoVector",
        )

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {"r": self.r, "phi": self.phi % (2 * numpy.pi) - numpy.pi},
            with_name="PolarTwoVector",
        )


@awkward.mixin_class(behavior)
class ThreeVector(TwoVector):
    """A cartesian 3-dimensional vector

    A heavy emphasis towards a momentum vector interpretation is assumed.
    This mixin class requires the parent class to provide items `x`, `y`, and `z`.
    """

    @property
    def pz(self):
        """Alias for `z`"""
        return self.z

    @property
    def rho2(self):
        """Squared `rho`"""
        return self.r2 + self.z ** 2

    @property
    def rho(self):
        r"""Distance from origin in 3D

        :math:`\sqrt{x^2+y^2+z^2} = \sqrt{r^2+z^2}`
        """
        return numpy.sqrt(self.rho2)

    @property
    def theta(self):
        r"""Inclination angle from XY plane

        :math:`\text{arctan2}(r, z)`
        """
        return numpy.arctan2(self.r, self.z)

    @property
    def p2(self):
        """Squared `p`"""
        return self.rho2

    @property
    def p(self):
        """Alias for `rho`"""
        return self.rho

    @awkward.mixin_class_method(numpy.absolute)
    def absolute(self):
        """Returns magnitude of the 3D vector

        Alias for `rho`
        """
        return self.p

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {"x": -self.x, "y": -self.y, "z": -self.z},
            with_name="ThreeVector",
        )

    @awkward.mixin_class_method(numpy.add, {"ThreeVector"})
    def add(self, other):
        """Add two vectors together elementwise using `x`, `y`, and `z` components"""
        return awkward.zip(
            {"x": self.x + other.x, "y": self.y + other.y, "z": self.z + other.z},
            with_name="ThreeVector",
        )

    @awkward.mixin_class_method(numpy.subtract, {"ThreeVector"})
    def subtract(self, other):
        """Subtract a vector from another elementwise using `x`, `y`, and `z` components"""
        return awkward.zip(
            {"x": self.x - other.x, "y": self.y - other.y, "z": self.z - other.z},
            with_name="ThreeVector",
        )

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `x`, `y`, and `z` components"""
        out = awkward.zip(
            {
                "x": awkward.sum(self.x, axis=axis),
                "y": awkward.sum(self.y, axis=axis),
                "z": awkward.sum(self.z, axis=axis),
            },
            with_name="ThreeVector",
            highlevel=False,
        )
        return awkward.Array(out, behavior=self.behavior)

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, and `z` components"""
        return awkward.zip(
            {"x": self.x * other, "y": self.y * other, "z": self.z * other},
            with_name="ThreeVector",
        )

    def dot(self, other):
        """Compute the dot product of two vectors"""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        """Compute the cross product of two vectors"""
        return awkward.zip(
            {
                "x": self.y * other.z - self.z * other.y,
                "y": self.z * other.x - self.x * other.z,
                "z": self.x * other.y - self.y * other.x,
            },
            with_name="ThreeVector",
        )

    @property
    def unit(self):
        """Unit vector, a vector of length 1 pointing in the same direction"""
        return self / self.rho


@awkward.mixin_class(behavior)
class SphericalThreeVector(ThreeVector, PolarTwoVector):
    """A spherical coordinate 3-dimensional vector

    This mixin class requires the parent class to provide items `rho`, `theta`, and `phi`.
    Some additional properties are overridden for performance
    """

    @property
    def r(self):
        r"""Distance from origin in XY plane

        :math:`\sqrt{x^2+y^2} = \rho \sin(\theta)`
        """
        return self.rho * numpy.sin(self.theta)

    @property
    def z(self):
        r"""Cartesian z value

        :math:`\rho \cos(\theta)`
        """
        return self.rho * numpy.cos(self.theta)

    @property
    def rho(self):
        r"""Distance from origin in 3D

        :math:`\sqrt{x^2+y^2+z^2} = \sqrt{r^2+z^2}`
        """
        return self["rho"]

    @property
    def theta(self):
        r"""Inclination angle from XY plane

        :math:`\text{arctan2}(r, z)`
        """
        return self["theta"]

    @property
    def p(self):
        """Alias for `rho`"""
        return self.rho

    @property
    def p2(self):
        """Squared `p`"""
        return self.rho ** 2

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, and `z` components

        In reality, this directly adjusts `r`, `theta` and `phi` for performance
        """
        return awkward.zip(
            {
                "rho": self.rho * abs(other),
                "theta": (numpy.sign(other) * self.theta + numpy.pi) % numpy.pi,
                "phi": self.phi % (2 * numpy.pi) - numpy.pi * (other < 0),
            },
            with_name="SphericalThreeVector",
        )

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {
                "rho": self.rho,
                "theta": (-self.theta + numpy.pi) % numpy.pi,
                "phi": self.phi % (2 * numpy.pi) - numpy.pi,
            },
            with_name="SphericalThreeVector",
        )


@awkward.mixin_class(behavior)
class LorentzVector(ThreeVector):
    """A cartesian Lorentz vector

    A heavy emphasis towards a momentum vector interpretation is assumed.
    (+, -, -, -) metric
    This mixin class requires the parent class to provide items `x`, `y`, `z`, and `t`.
    """

    @property
    def energy(self):
        """Alias for `t`"""
        return self.t

    @property
    def eta(self):
        r"""Pseudorapidity

        :math:`-\ln[\tan(\theta/2)] = \text{arcsinh}(z/r)`
        """
        return numpy.arcsinh(self.z / self.r)

    @property
    def mass2(self):
        """Squared `mass`"""
        return self.t ** 2 - self.p2

    @property
    def mass(self):
        r"""Invariant mass (+, -, -, -)

        :math:`\sqrt{t^2-x^2-y^2-z^2}`
        """
        return numpy.sqrt(self.mass2)

    @awkward.mixin_class_method(numpy.absolute)
    def absolute(self):
        """Magnitude of this Lorentz vector

        Alias for `mass`
        """
        return self.mass

    @awkward.mixin_class_method(numpy.add, {"LorentzVector"})
    def add(self, other):
        """Add two vectors together elementwise using `x`, `y`, `z`, and `t` components"""
        return awkward.zip(
            {
                "x": self.x + other.x,
                "y": self.y + other.y,
                "z": self.z + other.z,
                "t": self.t + other.t,
            },
            with_name="LorentzVector",
        )

    @awkward.mixin_class_method(numpy.subtract, {"LorentzVector"})
    def subtract(self, other):
        """Subtract a vector from another elementwise using `x`, `y`, `z`, and `t` components"""
        return awkward.zip(
            {
                "x": self.x - other.x,
                "y": self.y - other.y,
                "z": self.z - other.z,
                "t": self.t - other.t,
            },
            with_name="LorentzVector",
        )

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `x`, `y`, `z`, and `t` components"""
        out = awkward.zip(
            {
                "x": awkward.sum(self.x, axis=axis),
                "y": awkward.sum(self.y, axis=axis),
                "z": awkward.sum(self.z, axis=axis),
                "t": awkward.sum(self.t, axis=axis),
            },
            with_name="LorentzVector",
            highlevel=False,
        )
        return awkward.Array(out, behavior=self.behavior)

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, `z`, and `t` components"""
        return awkward.zip(
            {
                "x": self.x * other,
                "y": self.y * other,
                "z": self.z * other,
                "t": self.t * other,
            },
            with_name="LorentzVector",
        )

    def delta_r2(self, other):
        """Squared `delta_r`"""
        return (self.eta - other.eta) ** 2 + self.delta_phi(other) ** 2

    def delta_r(self, other):
        r"""Distance between two Lorentz vectors in (eta,phi) plane

        :math:`\sqrt{\Delta\eta^2 + \Delta\phi^2}`
        """
        return numpy.sqrt(self.delta_r2(other))

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {"x": -self.x, "y": -self.y, "z": -self.z, "t": -self.t},
            with_name="LorentzVector",
        )

    @property
    def pvec(self):
        """The `x`, `y` and `z` compontents as a `ThreeVector`"""
        return awkward.zip(
            {"x": self.x, "y": self.y, "z": self.z}, with_name="ThreeVector"
        )

    @property
    def boostvec(self):
        """The `x`, `y` and `z` compontents divided by `t` as a `ThreeVector`

        This can be used for boosting. For cases where `|t| <= rho`, this
        returns the unit vector.
        """
        rho = self.rho
        t = self.t
        with numpy.errstate(divide="ignore"):
            out = self.pvec * awkward.where(
                rho == 0, 0, awkward.where(abs(t) <= rho, 1 / rho, 1 / t)
            )
        return out

    def boost(self, other):
        """Apply a Lorentz boost given by the `ThreeVector` `other` and return it

        Note that this follows the convention that, for example in order to boost
        a vector into its own rest frame, one needs to use the negative of its `boostvec`
        """
        b2 = other.rho2
        gamma = (1 - b2) ** (-0.5)
        mask = b2 == 0
        b2 = awkward.where(mask, 1, b2)
        gamma2 = awkward.where(mask, 0, (gamma - 1) / b2)

        bp = self.dot(other)
        t = self.t
        v = gamma2 * bp * other + t * gamma * other

        out = awkward.zip(
            {
                "x": self.x + v.x,
                "y": self.y + v.y,
                "z": self.z + v.z,
                "t": gamma * (t + bp),
            },
            with_name="LorentzVector",
        )
        return out

    def metric_table(
        self, other, axis=1, metric=lambda a, b: a.delta_r(b), return_combinations=False
    ):
        """Return a list of a metric evaluated between this object and another.

        The two arrays should be broadcast-compatible on all axes other than the specified
        axis, which will be used to form a cartesian product. If axis=None, broadcast arrays directly.
        The return shape will be that of ``self`` with a new axis with shape of ``other`` appended
        at the specified axis depths.

        Parameters
        ----------
            other : awkward.Array
                Another array with same shape in all but ``axis``
            axis : int, optional
                The axis to form the cartesian product (default 1). If None, the metric
                is directly evaluated on the input arrays (i.e. they should broadcast)
            metric : callable
                A function of two arguments, returning a scalar. The default metric is `delta_r`.
            return_combinations : bool
                If True return the combinations of inputs as well as an unzipped tuple
        """
        if axis is None:
            a, b = self, other
        else:
            a, b = awkward.unzip(
                awkward.cartesian([self, other], axis=axis, nested=True)
            )
        mval = metric(a, b)
        if return_combinations:
            return mval, (a, b)
        return mval

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
            other : awkward.Array
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
        mval, (a, b) = self.metric_table(other, axis, metric, return_combinations=True)
        if axis is None:
            # NotImplementedError: awkward.firsts with axis=-1
            axis = other.layout.purelist_depth - 2
        mmin = awkward.argmin(mval, axis=axis + 1, keepdims=True)
        out = awkward.firsts(b[mmin], axis=axis + 1)
        metric = awkward.firsts(mval[mmin], axis=axis + 1)
        if threshold is not None:
            out = out.mask[metric <= threshold]
        if return_metric:
            return out, metric
        return out


@awkward.mixin_class(behavior)
class PtEtaPhiMLorentzVector(LorentzVector, SphericalThreeVector):
    """A Lorentz vector using pseudorapidity and mass

    This mixin class requires the parent class to provide items `pt`, `eta`, `phi`, and `mass`.
    Some additional properties are overridden for performance
    """

    @property
    def pt(self):
        """Alias for `r`"""
        return self["pt"]

    @property
    def eta(self):
        r"""Pseudorapidity

        :math:`-\ln\tan(\theta/2) = \text{arcsinh}(z/r)`
        """
        return self["eta"]

    @property
    def phi(self):
        r"""Azimuthal angle relative to X axis in XY plane

        :math:`\text{arctan2}(y, x)`
        """
        return self["phi"]

    @property
    def mass(self):
        r"""Invariant mass (+, -, -, -)

        :math:`\sqrt{t^2-x^2-y^2-z^2}`
        """
        return self["mass"]

    @property
    def rho(self):
        r"""Distance from origin in 3D

        :math:`\sqrt{x^2+y^2+z^2} = \sqrt{r^2+z^2}`
        """
        return self.pt * numpy.cosh(self.eta)

    @property
    def theta(self):
        r"""Inclination angle from XY plane

        :math:`\text{arctan2}(r, z) = 2\text{arctan}(e^{-\eta})`
        """
        return 2 * numpy.arctan(numpy.exp(-self.eta))

    @property
    def r(self):
        r"""Distance from origin in XY plane

        :math:`\sqrt{x^2+y^2} = \rho \sin(\theta)`
        """
        return self.pt

    @property
    def z(self):
        r"""Cartesian z value

        :math:`\rho \cos(\theta) = r \sinh(\eta)`
        """
        return self.pt * numpy.sinh(self.eta)

    @property
    def t(self):
        r"""Cartesian time component

        :math:`\sqrt{\rho^2+m^2}`
        """
        return numpy.hypot(self.rho, self.mass)

    @property
    def rho2(self):
        """Squared `rho`"""
        return self.rho ** 2

    @property
    def mass2(self):
        """Squared `mass`"""
        return self.mass ** 2

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, `z`, and `t` components

        In reality, this directly adjusts `pt`, `eta`, `phi` and `mass` for performance
        """
        absother = abs(other)
        return awkward.zip(
            {
                "pt": self.pt * absother,
                "eta": self.eta * numpy.sign(other),
                "phi": self.phi % (2 * numpy.pi) - (numpy.pi * (other < 0)),
                "mass": self.mass * absother,
            },
            with_name="PtEtaPhiMLorentzVector",
        )

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {
                "pt": self.pt,
                "eta": -self.eta,
                "phi": self.phi % (2 * numpy.pi) - numpy.pi,
                "mass": self.mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )


@awkward.mixin_class(behavior)
class PtEtaPhiELorentzVector(LorentzVector, SphericalThreeVector):
    """A Lorentz vector using pseudorapidity and energy

    This mixin class requires the parent class to provide items `pt`, `eta`, `phi`, and `energy`.
    Some additional properties are overridden for performance
    """

    @property
    def pt(self):
        """Alias for `r`"""
        return self["pt"]

    @property
    def eta(self):
        r"""Pseudorapidity

        :math:`-\ln\tan(\theta/2) = \text{arcsinh}(z/r)`
        """
        return self["eta"]

    @property
    def phi(self):
        r"""Azimuthal angle relative to X axis in XY plane

        :math:`\text{arctan2}(y, x)`
        """
        return self["phi"]

    @property
    def energy(self):
        """Alias for `t`"""
        return self["energy"]

    @property
    def t(self):
        r"""Cartesian time component

        :math:`\sqrt{\rho^2+m^2}`
        """
        return self["energy"]

    @property
    def rho(self):
        r"""Distance from origin in 3D

        :math:`\sqrt{x^2+y^2+z^2} = \sqrt{r^2+z^2}`
        """
        return self.pt * numpy.cosh(self.eta)

    @property
    def theta(self):
        r"""Inclination angle from XY plane

        :math:`\text{arctan2}(r, z) = 2\text{arctan}(e^{-\eta})`
        """
        return 2 * numpy.arctan(numpy.exp(-self.eta))

    @property
    def r(self):
        r"""Distance from origin in XY plane

        :math:`\sqrt{x^2+y^2} = \rho \sin(\theta)`
        """
        return self.pt

    @property
    def z(self):
        r"""Cartesian z value

        :math:`r \sinh(\eta)`
        """
        return self.pt * numpy.sinh(self.eta)

    @property
    def rho2(self):
        """Squared `rho`"""
        return self.rho ** 2

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, `z`, and `t` components

        In reality, this directly adjusts `pt`, `eta`, `phi` and `energy` for performance
        """
        return awkward.zip(
            {
                "pt": self.pt * abs(other),
                "eta": self.eta * numpy.sign(other),
                "phi": self.phi % (2 * numpy.pi) - (numpy.pi * (other < 0)),
                "energy": self.energy * other,
            },
            with_name="PtEtaPhiELorentzVector",
        )

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {
                "pt": self.pt,
                "eta": -self.eta,
                "phi": self.phi % (2 * numpy.pi) - numpy.pi,
                "energy": -self.energy,
            },
            with_name="PtEtaPhiELorentzVector",
        )


__all__ = [
    "TwoVector",
    "PolarTwoVector",
    "ThreeVector",
    "SphericalThreeVector",
    "LorentzVector",
    "PtEtaPhiMLorentzVector",
    "PtEtaPhiELorentzVector",
]
