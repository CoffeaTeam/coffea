from __future__ import print_function, division
from coffea import processor
import numpy as np


def test_accumulators():
    a = processor.accumulator(0.)
    a += 3.
    a += processor.accumulator(2)
    assert a.value == 5.
    assert a.identity().value == 0.

    a = processor.accumulator(np.array([0.]))
    a += 3.
    a += processor.accumulator(2)
    assert a.value == np.array([5.])
    assert a.identity().value == np.array([0.])

    b = processor.set_accumulator({'apples', 'oranges'})
    b += {'pears'}
    b += 'grapes'
    assert b == {'apples', 'oranges', 'pears', 'grapes'}

    c = processor.dict_accumulator({'num': a, 'fruit': b})
    c['num'] += 2.
    c += processor.dict_accumulator({
        'num2': processor.accumulator(0),
        'fruit': processor.set_accumulator({'apples', 'cherries'}),
    })
    assert c['num2'].value == 0
    assert c['num'].value == 7.
    assert c['fruit'] == {'apples', 'oranges', 'pears', 'grapes', 'cherries'}

    d = processor.defaultdict_accumulator(lambda: processor.accumulator(0.))
    d['x'] = processor.accumulator(0.)
    d['x'] += 4.
    d['y'] += 5.
    d['z'] += d['x']
    d['x'] += d['y']
    assert d['x'].value == 9.
    assert d['y'].value == 5.
    assert d['z'].value == 4.
    assert d['w'].value == 0.

    e = d + c

    f = processor.defaultdict_accumulator(lambda: 2.)
    f['x'] += 4.
    assert f['x'] == 6.

    f += f
    assert f['x'] == 12.
    assert f['y'] == 2.
