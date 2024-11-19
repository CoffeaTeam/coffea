import numpy as np
from dummy_distributions import dummy_jagged_eta_pt

import pytest


def test_weights():
    from coffea.analysis_tools import Weights

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    scale_central = np.random.normal(loc=1.0, scale=0.01, size=counts.size)
    scale_up = scale_central * 1.10
    scale_down = scale_central * 0.95
    scale_up_shift = 0.10 * scale_central
    scale_down_shift = 0.05 * scale_central

    weight = Weights(counts.size)
    weight.add("test", scale_central, weightUp=scale_up, weightDown=scale_down)
    weight.add(
        "testShift",
        scale_central,
        weightUp=scale_up_shift,
        weightDown=scale_down_shift,
        shift=True,
    )

    var_names = weight.variations
    expected_names = ["testShiftUp", "testShiftDown", "testUp", "testDown"]
    for name in expected_names:
        assert name in var_names

    test_central = weight.weight()
    exp_weight = scale_central * scale_central

    assert np.all(np.abs(test_central - (exp_weight)) < 1e-6)

    test_up = weight.weight("testUp")
    exp_up = scale_central * scale_central * 1.10

    assert np.all(np.abs(test_up - (exp_up)) < 1e-6)

    test_down = weight.weight("testDown")
    exp_down = scale_central * scale_central * 0.95

    assert np.all(np.abs(test_down - (exp_down)) < 1e-6)

    test_shift_up = weight.weight("testUp")

    assert np.all(np.abs(test_shift_up - (exp_up)) < 1e-6)

    test_shift_down = weight.weight("testDown")

    assert np.all(np.abs(test_shift_down - (exp_down)) < 1e-6)


def test_weights_multivariation():
    from coffea.analysis_tools import Weights

    counts, test_eta, test_pt = dummy_jagged_eta_pt()
    scale_central = np.random.normal(loc=1.0, scale=0.01, size=counts.size)
    scale_up = scale_central * 1.10
    scale_down = scale_central * 0.95
    scale_up_2 = scale_central * 1.2
    scale_down_2 = scale_central * 0.90

    weight = Weights(counts.size)
    weight.add_multivariation(
        "test",
        scale_central,
        modifierNames=["A", "B"],
        weightsUp=[scale_up, scale_up_2],
        weightsDown=[scale_down, scale_down_2],
    )

    var_names = weight.variations
    expected_names = ["test_AUp", "test_ADown", "test_BUp", "test_BDown"]
    for name in expected_names:
        assert name in var_names

    test_central = weight.weight()
    exp_weight = scale_central

    assert np.all(np.abs(test_central - (exp_weight)) < 1e-6)

    test_up = weight.weight("test_AUp")
    exp_up = scale_central * 1.10

    assert np.all(np.abs(test_up - (exp_up)) < 1e-6)

    test_down = weight.weight("test_ADown")
    exp_down = scale_central * 0.95

    assert np.all(np.abs(test_down - (exp_down)) < 1e-6)

    test_up_2 = weight.weight("test_BUp")
    exp_up = scale_central * 1.2

    assert np.all(np.abs(test_up_2 - (exp_up)) < 1e-6)

    test_down_2 = weight.weight("test_BDown")
    exp_down = scale_central * 0.90

    assert np.all(np.abs(test_down_2 - (exp_down)) < 1e-6)


def test_weights_partial():
    from coffea.analysis_tools import Weights

    counts, _, _ = dummy_jagged_eta_pt()
    w1 = np.random.normal(loc=1.0, scale=0.01, size=counts.size)
    w2 = np.random.normal(loc=1.3, scale=0.05, size=counts.size)

    weights = Weights(counts.size, storeIndividual=True)
    weights.add("w1", w1)
    weights.add("w2", w2)

    test_exclude_none = weights.weight()
    assert np.all(np.abs(test_exclude_none - w1 * w2) < 1e-6)

    test_exclude1 = weights.partial_weight(exclude=["w1"])
    assert np.all(np.abs(test_exclude1 - w2) < 1e-6)

    test_include1 = weights.partial_weight(include=["w1"])
    assert np.all(np.abs(test_include1 - w1) < 1e-6)

    test_exclude2 = weights.partial_weight(exclude=["w2"])
    assert np.all(np.abs(test_exclude2 - w1) < 1e-6)

    test_include2 = weights.partial_weight(include=["w2"])
    assert np.all(np.abs(test_include2 - w2) < 1e-6)

    test_include_both = weights.partial_weight(include=["w1", "w2"])
    assert np.all(np.abs(test_include_both - w1 * w2) < 1e-6)

    # Check that exception is thrown if arguments are incompatible
    error_raised = False
    try:
        weights.partial_weight(exclude=["w1"], include=["w2"])
    except ValueError:
        error_raised = True
    assert error_raised

    error_raised = False
    try:
        weights.partial_weight()
    except ValueError:
        error_raised = True
    assert error_raised

    # Check that exception is thrown if individual weights
    # are not saved from the start
    weights = Weights(counts.size, storeIndividual=False)
    weights.add("w1", w1)
    weights.add("w2", w2)

    error_raised = False
    try:
        weights.partial_weight(exclude=["test"], include=["test"])
    except ValueError:
        error_raised = True
    assert error_raised


def test_packed_selection():
    from coffea.analysis_tools import PackedSelection

    sel = PackedSelection()

    shape = (10,)
    all_true = np.full(shape=shape, fill_value=True, dtype=bool)
    all_false = np.full(shape=shape, fill_value=False, dtype=bool)
    fizz = np.arange(shape[0]) % 3 == 0
    buzz = np.arange(shape[0]) % 5 == 0
    ones = np.ones(shape=shape, dtype=np.uint64)
    wrong_shape = ones = np.ones(shape=(shape[0] - 5,), dtype=bool)

    sel.add("all_true", all_true)
    sel.add("all_false", all_false)
    sel.add("fizz", fizz)
    sel.add("buzz", buzz)

    assert np.all(sel.require(all_true=True, all_false=False) == all_true)
    # allow truthy values
    assert np.all(sel.require(all_true=1, all_false=0) == all_true)
    assert np.all(sel.all("all_true", "all_false") == all_false)
    assert np.all(sel.any("all_true", "all_false") == all_true)
    assert np.all(
        sel.all("fizz", "buzz")
        == np.array(
            [True, False, False, False, False, False, False, False, False, False]
        )
    )
    assert np.all(
        sel.any("fizz", "buzz")
        == np.array([True, False, False, True, False, True, True, False, False, True])
    )

    with pytest.raises(ValueError):
        sel.add("wrong_shape", wrong_shape)

    with pytest.raises(ValueError):
        sel.add("ones", ones)

    with pytest.raises(RuntimeError):
        overpack = PackedSelection()
        for i in range(65):
            overpack.add("sel_%d", all_true)
