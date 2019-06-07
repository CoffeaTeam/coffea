from coffea.util import load, save
from coffea.processor.test_items import NanoTestProcessor
import os

def test_loadsave():
    filename = 'testprocessor.coffea'
    try:
        aprocessor = NanoTestProcessor()
        save(aprocessor, filename)
        newprocessor = load(filename)
        assert 'pt' in newprocessor.accumulator
        assert newprocessor.accumulator['pt'].compatible(aprocessor.accumulator['pt'])
    finally:
        if os.path.exists(filename):
            os.remove(filename)
