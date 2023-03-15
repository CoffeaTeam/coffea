import awkward as ak
import numpy as np
import pytest

from coffea.nanoevents.methods import vector

ATOL = 1e-8


def record_arrays_equal(a, b):
    return (ak.fields(a) == ak.fields(b)) and all(
        ak.all(a[f] == b[f]) for f in ak.fields(a)
    )


def test_two_vector():
    a = ak.zip(
        {"x": [[1, 2], [], [3], [4]], "y": [[5, 6], [], [7], [8]]},
        with_name="TwoVector",
        behavior=vector.behavior,
    )
    b = ak.zip(
        {"x": [[11, 12], [], [13], [14]], "y": [[15, 16], [], [17], [18]]},
        with_name="TwoVector",
        behavior=vector.behavior,
    )

    assert record_arrays_equal(
        -a, ak.zip({"x": [[-1, -2], [], [-3], [-4]], "y": [[-5, -6], [], [-7], [-8]]})
    )

    assert record_arrays_equal(
        a + b,
        ak.zip({"x": [[12, 14], [], [16], [18]], "y": [[20, 22], [], [24], [26]]}),
    )
    assert record_arrays_equal(
        a - b,
        ak.zip(
            {"x": [[-10, -10], [], [-10], [-10]], "y": [[-10, -10], [], [-10], [-10]]}
        ),
    )

    assert record_arrays_equal(
        a * 2, ak.zip({"x": [[2, 4], [], [6], [8]], "y": [[10, 12], [], [14], [16]]})
    )
    assert record_arrays_equal(
        a / 2,
        ak.zip({"x": [[0.5, 1], [], [1.5], [2]], "y": [[2.5, 3], [], [3.5], [4]]}),
    )

    assert record_arrays_equal(a.dot(b), ak.Array([[86, 120], [], [158], [200]]))
    assert record_arrays_equal(b.dot(a), ak.Array([[86, 120], [], [158], [200]]))

    assert ak.all(abs(a.unit.r - 1) < ATOL)
    assert ak.all(abs(a.unit.phi - a.phi) < ATOL)


def test_polar_two_vector():
    a = ak.zip(
        {
            "r": [[1, 2], [], [3], [4]],
            "phi": [[0.3, 0.4], [], [0.5], [0.6]],
        },
        with_name="PolarTwoVector",
        behavior=vector.behavior,
    )

    assert record_arrays_equal(
        a * 2,
        ak.zip({"r": [[2, 4], [], [6], [8]], "phi": [[0.3, 0.4], [], [0.5], [0.6]]}),
    )
    assert ak.all((a * (-2)).r == [[2, 4], [], [6], [8]])
    assert ak.all(
        (a * (-2)).phi
        - ak.Array(
            [[-2.8415926535, -2.7415926535], [], [-2.6415926535], [-2.5415926535]]
        )
        < ATOL
    )
    assert record_arrays_equal(
        a / 2,
        ak.zip(
            {"r": [[0.5, 1], [], [1.5], [2]], "phi": [[0.3, 0.4], [], [0.5], [0.6]]}
        ),
    )

    assert ak.all(abs((-a).x + a.x) < ATOL)
    assert ak.all(abs((-a).y + a.y) < ATOL)
    assert record_arrays_equal(a * (-1), -a)

    assert ak.all(a.unit.phi == a.phi)


def test_three_vector():
    a = ak.zip(
        {
            "x": [[1, 2], [], [3], [4]],
            "y": [[5, 6], [], [7], [8]],
            "z": [[9, 10], [], [11], [12]],
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )
    b = ak.zip(
        {
            "x": [[4, 1], [], [10], [11]],
            "y": [[17, 7], [], [11], [6]],
            "z": [[9, 11], [], [5], [16]],
        },
        with_name="ThreeVector",
        behavior=vector.behavior,
    )

    assert record_arrays_equal(
        -a,
        ak.zip(
            {
                "x": [[-1, -2], [], [-3], [-4]],
                "y": [[-5, -6], [], [-7], [-8]],
                "z": [[-9, -10], [], [-11], [-12]],
            }
        ),
    )

    assert record_arrays_equal(
        a + b,
        ak.zip(
            {
                "x": [[5, 3], [], [13], [15]],
                "y": [[22, 13], [], [18], [14]],
                "z": [[18, 21], [], [16], [28]],
            }
        ),
    )
    assert record_arrays_equal(
        a - b,
        ak.zip(
            {
                "x": [[-3, 1], [], [-7], [-7]],
                "y": [[-12, -1], [], [-4], [2]],
                "z": [[0, -1], [], [6], [-4]],
            }
        ),
    )
    assert record_arrays_equal(
        b - a,
        ak.zip(
            {
                "x": [[3, -1], [], [7], [7]],
                "y": [[12, 1], [], [4], [-2]],
                "z": [[0, 1], [], [-6], [4]],
            }
        ),
    )

    assert record_arrays_equal(
        a * 2,
        ak.zip(
            {
                "x": [[2, 4], [], [6], [8]],
                "y": [[10, 12], [], [14], [16]],
                "z": [[18, 20], [], [22], [24]],
            }
        ),
    )
    assert record_arrays_equal(
        a / 2,
        ak.zip(
            {
                "x": [[0.5, 1], [], [1.5], [2]],
                "y": [[2.5, 3], [], [3.5], [4]],
                "z": [[4.5, 5], [], [5.5], [6]],
            }
        ),
    )

    assert ak.all(a.dot(b) == ak.Array([[170, 154], [], [162], [284]]))
    assert ak.all(b.dot(a) == ak.Array([[170, 154], [], [162], [284]]))

    assert record_arrays_equal(
        a.cross(b),
        ak.zip(
            {
                "x": [[-108, -4], [], [-86], [56]],
                "y": [[27, -12], [], [95], [68]],
                "z": [[-3, 8], [], [-37], [-64]],
            }
        ),
    )
    assert record_arrays_equal(
        b.cross(a),
        ak.zip(
            {
                "x": [[108, 4], [], [86], [-56]],
                "y": [[-27, 12], [], [-95], [-68]],
                "z": [[3, -8], [], [37], [64]],
            }
        ),
    )

    assert ak.all(abs(a.unit.rho - 1) < ATOL)
    assert ak.all(abs(a.unit.phi - a.phi) < ATOL)


def test_spherical_three_vector():
    a = ak.zip(
        {
            "rho": [[1.0, 2.0], [], [3.0], [4.0]],
            "theta": [[1.2, 0.7], [], [1.8], [1.9]],
            "phi": [[0.3, 0.4], [], [0.5], [0.6]],
        },
        with_name="SphericalThreeVector",
        behavior=vector.behavior,
    )

    assert ak.all(abs((-a).x + a.x) < ATOL)
    assert ak.all(abs((-a).y + a.y) < ATOL)
    assert ak.all(abs((-a).z + a.z) < ATOL)
    assert record_arrays_equal(a * (-1), -a)


def test_lorentz_vector():
    a = ak.zip(
        {
            "x": [[1, 2], [], [3], [4]],
            "y": [[5, 6], [], [7], [8]],
            "z": [[9, 10], [], [11], [12]],
            "t": [[50, 51], [], [52], [53]],
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )
    b = ak.zip(
        {
            "x": [[4, 1], [], [10], [11]],
            "y": [[17, 7], [], [11], [6]],
            "z": [[9, 11], [], [5], [16]],
            "t": [[60, 61], [], [62], [63]],
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )

    assert record_arrays_equal(
        -a,
        ak.zip(
            {
                "x": [[-1, -2], [], [-3], [-4]],
                "y": [[-5, -6], [], [-7], [-8]],
                "z": [[-9, -10], [], [-11], [-12]],
                "t": [[-50, -51], [], [-52], [-53]],
            }
        ),
    )

    assert record_arrays_equal(
        a + b,
        ak.zip(
            {
                "x": [[5, 3], [], [13], [15]],
                "y": [[22, 13], [], [18], [14]],
                "z": [[18, 21], [], [16], [28]],
                "t": [[110, 112], [], [114], [116]],
            }
        ),
    )
    assert record_arrays_equal(
        a - b,
        ak.zip(
            {
                "x": [[-3, 1], [], [-7], [-7]],
                "y": [[-12, -1], [], [-4], [2]],
                "z": [[0, -1], [], [6], [-4]],
                "t": [[-10, -10], [], [-10], [-10]],
            }
        ),
    )

    assert record_arrays_equal(
        a * 2,
        ak.zip(
            {
                "x": [[2, 4], [], [6], [8]],
                "y": [[10, 12], [], [14], [16]],
                "z": [[18, 20], [], [22], [24]],
                "t": [[100, 102], [], [104], [106]],
            }
        ),
    )
    assert record_arrays_equal(
        a / 2,
        ak.zip(
            {
                "x": [[0.5, 1], [], [1.5], [2]],
                "y": [[2.5, 3], [], [3.5], [4]],
                "z": [[4.5, 5], [], [5.5], [6]],
                "t": [[25, 25.5], [], [26], [26.5]],
            }
        ),
    )

    assert record_arrays_equal(
        a.pvec,
        ak.zip(
            {
                "x": [[1, 2], [], [3], [4]],
                "y": [[5, 6], [], [7], [8]],
                "z": [[9, 10], [], [11], [12]],
            }
        ),
    )

    boosted = a.boost(-a.boostvec)
    assert ak.all(abs(boosted.x) < ATOL)
    assert ak.all(abs(boosted.y) < ATOL)
    assert ak.all(abs(boosted.z) < ATOL)


def test_pt_eta_phi_m_lorentz_vector():
    a = ak.zip(
        {
            "pt": [[1, 2], [], [3], [4]],
            "eta": [[1.2, 1.4], [], [1.6], [3.4]],
            "phi": [[0.3, 0.4], [], [0.5], [0.6]],
            "mass": [[0.5, 0.9], [], [1.3], [4.5]],
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )
    a = ak.Array(a, behavior=vector.behavior)

    assert ak.all((a * (-2)).pt == ak.Array([[2, 4], [], [6], [8]]))
    assert ak.all(
        (a * (-2)).theta
        - ak.Array(
            [[2.556488570968, 2.65804615357], [], [2.74315571762], [3.07487087733]]
        )
        < ATOL
    )
    assert ak.all(
        (a * (-2)).phi
        - ak.Array(
            [[-2.8415926535, -2.7415926535], [], [-2.6415926535], [-2.5415926535]]
        )
        < ATOL
    )
    assert record_arrays_equal(
        a / 2,
        ak.zip(
            {
                "pt": [[0.5, 1], [], [1.5], [2]],
                "eta": [[1.2, 1.4], [], [1.6], [3.4]],
                "phi": [[0.3, 0.4], [], [0.5], [0.6]],
                "mass": [[0.25, 0.45], [], [0.65], [2.25]],
            }
        ),
    )
    assert record_arrays_equal(a * (-1), -a)

    boosted = a.boost(-a.boostvec)
    assert ak.all(abs(boosted.x) < ATOL)
    assert ak.all(abs(boosted.y) < ATOL)
    assert ak.all(abs(boosted.z) < ATOL)


def test_pt_eta_phi_e_lorentz_vector():
    a = ak.zip(
        {
            "pt": [[1, 2], [], [3], [4]],
            "eta": [[1.2, 1.4], [], [1.6], [3.4]],
            "phi": [[0.3, 0.4], [], [0.5], [0.6]],
            "energy": [[50, 51], [], [52], [60]],
        },
        with_name="PtEtaPhiELorentzVector",
        behavior=vector.behavior,
    )

    assert ak.all((a * (-2)).pt == ak.Array([[2, 4], [], [6], [8]]))
    assert ak.all(
        (a * (-2)).theta
        - ak.Array(
            [[2.556488570968, 2.65804615357], [], [2.74315571762], [3.07487087733]]
        )
        < ATOL
    )
    assert ak.all(
        (a * (-2)).phi
        - ak.Array(
            [[-2.8415926535, -2.7415926535], [], [-2.6415926535], [-2.5415926535]]
        )
        < ATOL
    )
    assert record_arrays_equal(
        a / 2,
        ak.zip(
            {
                "pt": [[0.5, 1], [], [1.5], [2]],
                "eta": [[1.2, 1.4], [], [1.6], [3.4]],
                "phi": [[0.3, 0.4], [], [0.5], [0.6]],
                "energy": [[25, 25.5], [], [26], [30]],
            }
        ),
    )
    assert record_arrays_equal(a * (-1), -a)

    boosted = a.boost(-a.boostvec)
    assert ak.all(abs(boosted.x) < ATOL)
    assert ak.all(abs(boosted.y) < ATOL)
    assert ak.all(abs(boosted.z) < ATOL)


@pytest.mark.parametrize("a_dtype", ["i4", "f4", "f8"])
@pytest.mark.parametrize("b_dtype", ["i4", "f4", "f8"])
def test_lorentz_vector_numba(a_dtype, b_dtype):
    a = ak.zip(
        {
            "x": np.array([1, 2, 3, 4], dtype=a_dtype),
            "y": np.array([5, 6, 7, 8], dtype=a_dtype),
            "z": np.array([9, 10, 11, 12], dtype=a_dtype),
            "t": np.array([50, 51, 52, 53], dtype=b_dtype),  # b on purpose
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )
    b = ak.zip(
        {
            "x": np.array([4, 1, 10, 11], dtype=b_dtype),
            "y": np.array([17, 7, 11, 6], dtype=b_dtype),
            "z": np.array([9, 11, 5, 16], dtype=b_dtype),
            "t": np.array([60, 61, 62, 63], dtype=b_dtype),
        },
        with_name="LorentzVector",
        behavior=vector.behavior,
    )
    assert pytest.approx(a.mass) == [
        48.91829923454004,
        49.60846701924985,
        50.24937810560445,
        50.84289527554464,
    ]
    assert pytest.approx((a + b).mass) == [
        106.14612569472331,
        109.20164833920778,
        110.66616465749593,
        110.68423555321688,
    ]
    assert pytest.approx(a.delta_phi(b), abs=1e-7) == [
        0.03369510734601633,
        -0.1798534997924781,
        0.33292327383538156,
        0.6078019961139605,
    ]


@pytest.mark.parametrize(
    "lcoord", ["LorentzVector", "PtEtaPhiMLorentzVector", "PtEtaPhiELorentzVector"]
)
@pytest.mark.parametrize("threecoord", ["ThreeVector", "SphericalThreeVector"])
@pytest.mark.parametrize("twocoord", ["TwoVector", "PolarTwoVector"])
def test_inherited_method_transpose(lcoord, threecoord, twocoord):
    if lcoord == "LorentzVector":
        a = ak.zip(
            {
                "x": [10.0, 20.0, 30.0],
                "y": [-10.0, 20.0, 30.0],
                "z": [5.0, 10.0, 15.0],
                "t": [16.0, 31.0, 46.0],
            },
            with_name=lcoord,
            behavior=vector.behavior,
        )
    elif lcoord == "PtEtaPhiMLorentzVector":
        a = ak.zip(
            {
                "pt": [10.0, 20.0, 30.0],
                "eta": [0.0, 1.1, 2.2],
                "phi": [0.1, 0.9, -1.1],
                "mass": [1.0, 1.0, 1.0],
            },
            with_name=lcoord,
            behavior=vector.behavior,
        )
    elif lcoord == "PtEtaPhiELorentzVector":
        a = ak.zip(
            {
                "pt": [10.0, 20.0, 30.0],
                "eta": [0.0, 1.1, 2.2],
                "phi": [0.1, 0.9, -1.1],
                "energy": [11.0, 21.0, 31.0],
            },
            with_name=lcoord,
            behavior=vector.behavior,
        )
    if threecoord == "ThreeVector":
        b = ak.zip(
            {
                "x": [-10.0, 20.0, -30.0],
                "y": [-10.0, -20.0, 30.0],
                "z": [5.0, -10.0, 15.0],
            },
            with_name=threecoord,
            behavior=vector.behavior,
        )
    elif threecoord == "SphericalThreeVector":
        b = ak.zip(
            {
                "rho": [10.0, 20.0, 30.0],
                "theta": [0.3, 0.6, 1.1],
                "phi": [-3.0, 1.1, 0.2],
            },
            with_name=threecoord,
            behavior=vector.behavior,
        )
    if twocoord == "TwoVector":
        c = ak.zip(
            {"x": [-10.0, 13.0, 15.0], "y": [12.0, -4.0, 41.0]},
            with_name=twocoord,
            behavior=vector.behavior,
        )
    elif twocoord == "PolarTwoVector":
        c = ak.zip(
            {"r": [-10.0, 13.0, 15.0], "phi": [1.22, -1.0, 1.0]},
            with_name=twocoord,
            behavior=vector.behavior,
        )

    assert record_arrays_equal(a + b, b + a)
    assert record_arrays_equal(a + c, c + a)
    assert record_arrays_equal(b + c, c + b)

    assert record_arrays_equal(a.delta_phi(b), b.delta_phi(a))
    assert record_arrays_equal(a.delta_phi(c), c.delta_phi(a))
    assert record_arrays_equal(b.delta_phi(c), c.delta_phi(b))

    assert record_arrays_equal(a - b, -(b - a))
    assert record_arrays_equal(a - c, -(c - a))
    assert record_arrays_equal(b - c, -(c - b))
