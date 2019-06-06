from __future__ import print_function, division

from coffea.striped import ColumnGroup, PhysicalColumnGroup, jaggedFromColumnGroup
import uproot
import uproot_methods
from coffea.util import awkward
from coffea.util import numpy as np

from dummy_distributions import dummy_events

def test_striped():
    events = dummy_events()
    colgroup = ColumnGroup(events,"thing","p4","blah")
    physcolgroup = PhysicalColumnGroup(events,"thing","blah",p4="p4")
    physcolgroup2 = PhysicalColumnGroup(events,"thing","blah",
                                        pt="pt",eta="eta",phi="phi",mass="mass")
    physcolgroup3 = PhysicalColumnGroup(events,"thing","blah",
                                        px="px",py="py",pz="pz",energy="en")
    
    jagged = jaggedFromColumnGroup(colgroup)
    candjagged = jaggedFromColumnGroup(physcolgroup)
    candjagged2 = jaggedFromColumnGroup(physcolgroup2)
    candjagged3 = jaggedFromColumnGroup(physcolgroup3)
    
    assert not hasattr(jagged,'p4')
    assert hasattr(candjagged,'p4')
    assert hasattr(candjagged2,'p4')
    assert hasattr(candjagged3,'p4')
