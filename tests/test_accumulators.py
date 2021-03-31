import pytest
from collections import defaultdict
from coffea import processor
from functools import partial
import numpy as np


def test_accumulators():
    a = processor.value_accumulator(float)
    a += 3.
    assert a.value == 3.
    assert a.identity().value == 0.

    a = processor.value_accumulator(partial(np.array, [2.]))
    a += 3.
    assert np.array_equal(a.value, np.array([5.]))
    assert np.array_equal(a.identity().value, np.array([2.]))

    l = processor.list_accumulator(range(4))
    l += [3]
    l += processor.list_accumulator([1, 2])
    assert l == [0, 1, 2, 3, 3, 1, 2]

    b = processor.set_accumulator({'apples', 'oranges'})
    b += {'pears'}
    b += 'grapes'
    assert b == {'apples', 'oranges', 'pears', 'grapes'}

    c = processor.dict_accumulator({'num': a, 'fruit': b})
    c['num'] += 2.
    c += processor.dict_accumulator({
        'num2': processor.value_accumulator(int),
        'fruit': processor.set_accumulator({'apples', 'cherries'}),
    })
    assert c['num2'].value == 0
    assert np.array_equal(c['num'].value, np.array([7.]))
    assert c['fruit'] == {'apples', 'oranges', 'pears', 'grapes', 'cherries'}

    d = processor.defaultdict_accumulator(float)
    d['x'] = 0.
    d['x'] += 4.
    d['y'] += 5.
    d['z'] += d['x']
    d['x'] += d['y']
    assert d['x'] == 9.
    assert d['y'] == 5.
    assert d['z'] == 4.
    assert d['w'] == 0.

    e = d + c

    f = processor.defaultdict_accumulator(lambda: 2.)
    f['x'] += 4.
    assert f['x'] == 6.

    f += f
    assert f['x'] == 12.
    assert f['y'] == 2.

    a = processor.column_accumulator(np.arange(6).reshape(2,3))
    b = processor.column_accumulator(np.arange(12).reshape(4,3))
    a += b
    assert a.value.sum() == 81


def test_new_accumulators():
    a = processor.accumulate((0., 3.))
    assert a == 3.

    a = processor.accumulate((
        np.array([2.]),
        3.,
    ))
    assert np.array_equal(a, np.array([5.]))

    l = processor.accumulate((
        list(range(4)),
        [3],
        [1, 2],
    ))
    assert l == [0, 1, 2, 3, 3, 1, 2]

    b = processor.accumulate((
        {'apples', 'oranges'},
        {'pears'},
        {'grapes'},
    ))
    assert b == {'apples', 'oranges', 'pears', 'grapes'}

    c = processor.accumulate((
        {'num': a, 'fruit': b},
        {'num': 2.},
        {
            'num2': 0,
            'fruit': {'apples', 'cherries'},
        }
    ))
    assert c['num2'] == 0
    assert np.array_equal(c['num'], np.array([7.]))
    assert c['fruit'] == {'apples', 'oranges', 'pears', 'grapes', 'cherries'}

    d = processor.accumulate((
        defaultdict(float),
        {"x": 4., "y": 5.},
        {"z": 4., "x": 5.},
    ))
    assert d['x'] == 9.
    assert d['y'] == 5.
    assert d['z'] == 4.
    # this is different than old style!
    with pytest.raises(KeyError):
        d['w']

    e = processor.accumulate((d, c))

    f = processor.accumulate((
        defaultdict(lambda: 2.),
        defaultdict(lambda: 2, {"x": 4.}),
    ))
    assert f['x'] == 4.
    assert f['y'] == 2.

    # this is different than old style!
    f = processor.accumulate([f], f)
    assert f['x'] == 8.
    assert f['y'] == 4.
    assert f['z'] == 2.

    a = processor.accumulate((
        processor.column_accumulator(np.arange(6).reshape(2,3)),
        processor.column_accumulator(np.arange(12).reshape(4,3)),
    ))
    assert a.value.sum() == 81


def test_accumulator_types():
    class MyDict(dict):
        pass

    out = processor.accumulate((
        {"x": 2},
        MyDict({"x": 3}),
    ))
    assert type(out) is dict

    with pytest.raises(ValueError):
        processor.accumulate((
            defaultdict(lambda: 2),
            MyDict({"x": 3}),
        ))

    out = processor.accumulate((
        MyDict({"x": 3}),
        {"x": 2},
    ))
    assert type(out) is dict

    with pytest.raises(ValueError):
        processor.accumulate((
            MyDict({"x": 3}),
            defaultdict(lambda: 2),
        ))
