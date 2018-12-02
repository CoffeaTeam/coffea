from __future__ import print_function, division

from fnal_column_analysis_tools.striped import ColumnGroup, PhysicalColumnGroup, jaggedFromColumnGroup
import uproot
import uproot_methods
import awkward
import numpy as np

from dummy_distributions import dummy_events

def test_striped():
    events = dummy_events()
    colgroup = ColumnGroup(events,"thing","p4","blah")
    physcolgroup = PhysicalColumnGroup(events,"thing","p4","blah")

    jagged = jaggedFromColumnGroup(colgroup)
    candjagged = jaggedFromColumnGroup(physcolgroup)
    
    assert not hasattr(jagged,'p4')
    assert hasattr(candjagged,'p4')
