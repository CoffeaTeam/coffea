"""2D, 3D, and Lorentz vector class mixins

These mixins will eventually be superseded by the `vector <https://github.com/scikit-hep/vector>`__ library,
which will hopefully be feature-compatible. The 2D vector provides cartesian and polar coordinate attributes,
where ``r`` represents the polar distance from the origin..  The 3D vector provides cartesian and spherical coordinates,
where ``rho`` represents the 3D distance from the origin and ``r`` is the axial distance from the z axis, so that it can
subclass the 2D vector. The Lorentz vector also subclasses the 3D vector, adding ``t`` as the fourth
cartesian coordinate. Aliases typical of momentum vectors are also provided.

A small example::

    import numpy as np
    import awkward as ak
    from coffea.nanoevents.methods import vector

    n = 1000

    vec = ak.zip(
        {
            "x": np.random.normal(size=n),
            "y": np.random.normal(size=n),
            "z": np.random.normal(size=n),
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )

    vec4 = ak.zip(
        {
            "pt": vec.r,
            "eta": -np.log(np.tan(vec.theta/2)),
            "phi": vec.phi,
            "mass": np.full(n, 1.),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

    assert np.allclose(np.array(vec4.x), np.array(vec.x))
    assert np.allclose(np.array(vec4.y), np.array(vec.y))
    assert np.allclose(np.array(vec4.z), np.array(vec.z))
    assert np.allclose(np.array(abs(2*vec + vec4) / abs(vec)), 3)

"""

import numbers
from datetime import datetime

import awkward
import numba
import numpy
import pytz
import vector
from dask_awkward import dask_method
from vector.backends.awkward import (
    MomentumAwkward2D,
    MomentumAwkward3D,
    MomentumAwkward4D,
)

from coffea.util import deprecate

_cst = pytz.timezone("US/Central")
_depttime = _cst.localize(datetime(2024, 6, 30, 11, 59, 59))
deprecate(
    (
        "coffea.nanoevents.methods.vector will be removed and replaced with scikit-hep vector. "
        "Nanoevents schemas internal to coffea will be migrated. "
        "Otherwise please consider using that package!"
    ),
    version="2024.7.0",
    date=str(_depttime),
    category=FutureWarning,
)


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32, numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64, numba.float64, numba.float64),
    ]
)
def _mass2_kernel(t, x, y, z):
    return t * t - x * x - y * y - z * z


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64),
    ]
)
def delta_phi(a, b):
    """Compute difference in angle given two angles a and b

    Returns a value within [-pi, pi)
    """
    return (a - b + numpy.pi) % (2 * numpy.pi) - numpy.pi


@numba.vectorize(
    [
        numba.float32(numba.float32, numba.float32, numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64, numba.float64, numba.float64),
    ]
)
def delta_r(eta1, phi1, eta2, phi2):
    r"""Distance in (eta,phi) plane given two pairs of (eta,phi)

    :math:`\sqrt{\Delta\eta^2 + \Delta\phi^2}`
    """
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return numpy.hypot(deta, dphi)


behavior = {}
behavior.update(vector.backends.awkward.behavior)


class TwoVectorArray:
    pass


class PolarTwoVectorArray:
    pass


class ThreeVectorArray:
    pass


class SphericalThreeVectorArray:
    pass


class LorentzVectorArray:
    pass


class PtEtaPhiMLorentzVectorArray:
    pass


class PtEtaPhiELorentzVectorArray:
    pass


@awkward.mixin_class(behavior)
class TwoVector(MomentumAwkward2D):
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
        return self.rho

    @property
    def r2(self):
        """Squared `r`"""
        return self.rho2

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
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.add, {"TwoVector"})
    def add(self, other):
        """Add two vectors together elementwise using `x` and `y` components"""
        return awkward.zip(
            {"x": self.x + other.x, "y": self.y + other.y},
            with_name="TwoVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(
        numpy.subtract,
        {
            "TwoVector",
            "ThreeVector",
            "SphericalThreeVector",
            "LorentzVector",
            "PtEtaPhiMLorentzVector",
            "PtEtaPhiELorentzVector",
        },
        transpose=False,
    )
    def subtract(self, other):
        """Subtract a vector from another elementwise using `x` and `y` components"""
        return awkward.zip(
            {"x": self.x - other.x, "y": self.y - other.y},
            with_name="TwoVector",
            behavior=self.behavior,
        )

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `x` and `y` components"""
        return awkward.zip(
            {
                "x": awkward.sum(self.x, axis=axis),
                "y": awkward.sum(self.y, axis=axis),
            },
            with_name="TwoVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x` and `y` components"""
        return awkward.zip(
            {"x": self.x * other, "y": self.y * other},
            with_name="TwoVector",
            behavior=self.behavior,
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
        return self.deltaphi(other)

    @property
    def unit(self):
        """Unit vector, a vector of length 1 pointing in the same direction"""
        return self / self.r


@awkward.mixin_class(behavior)
class PolarTwoVector(TwoVector):
    """A polar coordinate 2-dimensional vector

    This mixin class requires the parent class to provide items `rho` and `phi`.
    Some additional properties are overridden for performance
    """

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using using `x` and `y` components

        In reality, this directly adjusts `r` and `phi` for performance
        """
        return awkward.zip(
            {
                "rho": self.r * abs(other),
                "phi": self.phi % (2 * numpy.pi) - (numpy.pi * (other < 0)),
            },
            with_name="PolarTwoVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {"rho": self.r, "phi": self.phi % (2 * numpy.pi) - numpy.pi},
            with_name="PolarTwoVector",
            behavior=self.behavior,
        )


@awkward.mixin_class(behavior)
class ThreeVector(MomentumAwkward3D):
    """A cartesian 3-dimensional vector

    A heavy emphasis towards a momentum vector interpretation is assumed.
    This mixin class requires the parent class to provide items `x`, `y`, and `z`.
    """

    @property
    def r(self):
        r"""Distance from origin in XY plane

        :math:`\sqrt{x^2+y^2}`
        """
        return self.rho

    @property
    def r2(self):
        """Squared `r`"""
        return self.rho2

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
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(
        numpy.add, {"ThreeVector", "TwoVector", "PolarTwoVector"}
    )
    def add(self, other):
        """Add two vectors together elementwise using `x`, `y`, and `z` components"""
        other_3d = other.to_Vector3D()
        return awkward.zip(
            {
                "x": self.x + other_3d.x,
                "y": self.y + other_3d.y,
                "z": self.z + other_3d.z,
            },
            with_name="ThreeVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.divide, {numbers.Number})
    def divide(self, other):
        """Divide this vector by a scalar elementwise using its cartesian components
        This is realized by using the multiplication functionality"""
        return self.multiply(1 / other)

    @awkward.mixin_class_method(
        numpy.subtract,
        {
            "TwoVector",
            "PolarTwoVector",
            "ThreeVector",
            "SphericalThreeVector",
            "LorentzVector",
            "PtEtaPhiMLorentzVector",
            "PtEtaPhiELorentzVector",
        },
        transpose=False,
    )
    def subtract(self, other):
        """Subtract a vector from another elementwise using `x`, `y`, and `z` components"""
        other_3d = other.to_Vector3D()

        return awkward.zip(
            {
                "x": self.x - other_3d.x,
                "y": self.y - other_3d.y,
                "z": self.z - other_3d.z,
            },
            with_name="ThreeVector",
            behavior=self.behavior,
        )

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `x`, `y`, and `z` components"""
        return awkward.zip(
            {
                "x": awkward.sum(self.x, axis=axis),
                "y": awkward.sum(self.y, axis=axis),
                "z": awkward.sum(self.z, axis=axis),
            },
            with_name="ThreeVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.multiply, {numbers.Number})
    def multiply(self, other):
        """Multiply this vector by a scalar elementwise using `x`, `y`, and `z` components"""
        return awkward.zip(
            {"x": self.x * other, "y": self.y * other, "z": self.z * other},
            with_name="ThreeVector",
            behavior=self.behavior,
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
            behavior=self.behavior,
        )

    def delta_phi(self, other):
        """Compute difference in angle between two vectors

        Returns a value within [-pi, pi)
        """
        return self.deltaphi(other)

    @property
    def unit(self):
        """Unit vector, a vector of length 1 pointing in the same direction"""
        return self / self.rho


@awkward.mixin_class(behavior)
class SphericalThreeVector(ThreeVector):
    """A spherical coordinate 3-dimensional vector

    This mixin class requires the parent class to provide items `rho`, `theta`, and `phi`.
    Some additional properties are overridden for performance
    """

    @property
    def r(self):
        r"""Distance from origin in XY plane

        :math:`\sqrt{x^2+y^2} = \rho \sin(\theta)`
        """
        return self.rho

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
            behavior=self.behavior,
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
            behavior=self.behavior,
        )


def _metric_table_core(a, b, axis, metric, return_combinations):
    if axis is None:
        a, b = a, b
    else:
        a, b = awkward.unzip(awkward.cartesian([a, b], axis=axis, nested=True))
    mval = metric(a, b)
    if return_combinations:
        return mval, (a, b)
    return mval


def _nearest_core(x, y, axis, metric, return_metric, threshold):
    mval, (a, b) = x.metric_table(y, axis, metric, return_combinations=True)
    if axis is None:
        # NotImplementedError: awkward.firsts with axis=-1
        axis = y.layout.purelist_depth - 2
    mmin = awkward.argmin(mval, axis=axis + 1, keepdims=True)
    out = awkward.firsts(b[mmin], axis=axis + 1)
    metric = awkward.firsts(mval[mmin], axis=axis + 1)
    if threshold is not None:
        out = awkward.mask(out, metric <= threshold)
    if return_metric:
        return out, metric
    return out


@awkward.mixin_class(behavior)
class LorentzVector(MomentumAwkward4D):
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
        return _mass2_kernel(self.t, self.x, self.y, self.z)

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

    @awkward.mixin_class_method(
        numpy.add,
        {
            "LorentzVector",
            "ThreeVector",
            "SphericalThreeVector",
            "TwoVector",
            "PolarTwoVector",
        },
    )
    def add(self, other):
        """Add two vectors together elementwise using `x`, `y`, `z`, and `t` components"""
        other_4d = other.to_Vector4D()
        return awkward.zip(
            {
                "x": self.x + other_4d.x,
                "y": self.y + other_4d.y,
                "z": self.z + other_4d.z,
                "t": self.t + other_4d.t,
            },
            with_name="LorentzVector",
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(
        numpy.subtract,
        {
            "LorentzVector",
            "ThreeVector",
            "SphericalThreeVector",
            "TwoVector",
            "PolarTwoVector",
        },
        transpose=False,
    )
    def subtract(self, other):
        """Subtract a vector from another elementwise using `x`, `y`, `z`, and `t` components"""
        other_4d = other.to_Vector4D()

        return awkward.zip(
            {
                "x": self.x - other_4d.x,
                "y": self.y - other_4d.y,
                "z": self.z - other_4d.z,
                "t": self.t - other_4d.t,
            },
            with_name="LorentzVector",
            behavior=self.behavior,
        )

    def sum(self, axis=-1):
        """Sum an array of vectors elementwise using `x`, `y`, `z`, and `t` components"""
        return awkward.zip(
            {
                "x": awkward.sum(self.x, axis=axis),
                "y": awkward.sum(self.y, axis=axis),
                "z": awkward.sum(self.z, axis=axis),
                "t": awkward.sum(self.t, axis=axis),
            },
            with_name="LorentzVector",
            behavior=self.behavior,
        )

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
            behavior=self.behavior,
        )

    @awkward.mixin_class_method(numpy.divide, {numbers.Number})
    def divide(self, other):
        """Divide this vector by a scalar elementwise using its cartesian components
        This is realized by using the multiplication functionality"""
        return self.multiply(1 / other)

    def delta_r2(self, other):
        """Squared `delta_r`"""
        return self.deltaR2(other)

    def delta_r(self, other):
        r"""Distance between two Lorentz vectors in (eta,phi) plane

        :math:`\sqrt{\Delta\eta^2 + \Delta\phi^2}`
        """
        return self.deltaR(other)

    def delta_phi(self, other):
        """Compute difference in angle between two vectors

        Returns a value within [-pi, pi)
        """
        return self.deltaphi(other)

    @awkward.mixin_class_method(numpy.negative)
    def negative(self):
        """Returns the negative of the vector"""
        return awkward.zip(
            {"x": -self.x, "y": -self.y, "z": -self.z, "t": -self.t},
            with_name="LorentzVector",
            behavior=self.behavior,
        )

    @property
    def pvec(self):
        """The `x`, `y` and `z` components as a `ThreeVector`"""
        return awkward.zip(
            {"x": self.x, "y": self.y, "z": self.z},
            with_name="ThreeVector",
            behavior=self.behavior,
        )

    @property
    def boostvec(self):
        """The `x`, `y` and `z` components divided by `t` as a `ThreeVector`

        This can be used for boosting. For cases where `|t| <= r`, this
        returns the unit vector.
        """
        return self.to_beta3()

    @dask_method
    def metric_table(
        self,
        other,
        axis=1,
        metric=lambda a, b: a.delta_r(b),
        return_combinations=False,
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
        return _metric_table_core(self, other, axis, metric, return_combinations)

    @metric_table.dask
    def metric_table(
        self,
        dask_array,
        other,
        axis=1,
        metric=lambda a, b: a.delta_r(b),
        return_combinations=False,
    ):
        return _metric_table_core(dask_array, other, axis, metric, return_combinations)

    @dask_method
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
        return _nearest_core(self, other, axis, metric, return_metric, threshold)

    @nearest.dask
    def nearest(
        self,
        dask_array,
        other,
        axis=1,
        metric=lambda a, b: a.delta_r(b),
        return_metric=False,
        threshold=None,
    ):
        return _nearest_core(dask_array, other, axis, metric, return_metric, threshold)


@awkward.mixin_class(behavior)
class PtEtaPhiMLorentzVector(LorentzVector):
    """A Lorentz vector using pseudorapidity and mass

    This mixin class requires the parent class to provide items `pt`, `eta`, `phi`, and `mass`.
    Some additional properties are overridden for performance
    """

    @property
    def E(self):
        """Alias for `t`"""
        return self.t

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
        return self.rho * self.rho

    @property
    def mass2(self):
        """Squared `mass`"""
        return self.mass * self.mass

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
            behavior=self.behavior,
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
            behavior=self.behavior,
        )


@awkward.mixin_class(behavior)
class PtEtaPhiELorentzVector(LorentzVector):
    """A Lorentz vector using pseudorapidity and energy

    This mixin class requires the parent class to provide items `pt`, `eta`, `phi`, and `energy`.
    Some additional properties are overridden for performance
    """

    @property
    def E(self):
        """Alias for `t`"""
        return self.t

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
        return self.rho * self.rho

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
            behavior=self.behavior,
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
            behavior=self.behavior,
        )


TwoVectorArray.ProjectionClass2D = TwoVectorArray
TwoVectorArray.ProjectionClass3D = ThreeVectorArray
TwoVectorArray.ProjectionClass4D = LorentzVectorArray

PolarTwoVectorArray.ProjectionClass2D = PolarTwoVectorArray
PolarTwoVectorArray.ProjectionClass3D = SphericalThreeVectorArray
PolarTwoVectorArray.ProjectionClass4D = LorentzVectorArray

ThreeVectorArray.ProjectionClass2D = TwoVectorArray
ThreeVectorArray.ProjectionClass3D = ThreeVectorArray
ThreeVectorArray.ProjectionClass4D = LorentzVectorArray

SphericalThreeVectorArray.ProjectionClass2D = TwoVectorArray
SphericalThreeVectorArray.ProjectionClass3D = SphericalThreeVectorArray
SphericalThreeVectorArray.ProjectionClass4D = LorentzVectorArray

LorentzVectorArray.ProjectionClass2D = TwoVectorArray
LorentzVectorArray.ProjectionClass3D = ThreeVectorArray
LorentzVectorArray.ProjectionClass4D = LorentzVectorArray

PtEtaPhiMLorentzVectorArray.ProjectionClass2D = TwoVectorArray
PtEtaPhiMLorentzVectorArray.ProjectionClass3D = ThreeVectorArray
PtEtaPhiMLorentzVectorArray.ProjectionClass4D = LorentzVectorArray

PtEtaPhiELorentzVectorArray.ProjectionClass2D = TwoVectorArray
PtEtaPhiELorentzVectorArray.ProjectionClass3D = ThreeVectorArray
PtEtaPhiELorentzVectorArray.ProjectionClass4D = LorentzVectorArray

__all__ = [
    "TwoVector",
    "PolarTwoVector",
    "ThreeVector",
    "SphericalThreeVector",
    "LorentzVector",
    "PtEtaPhiMLorentzVector",
    "PtEtaPhiELorentzVector",
]
