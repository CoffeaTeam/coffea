from coffea.util import load, save
from coffea.processor.test_items import NanoEventsProcessor
import os


def test_loadsave():
    filename = "testprocessor.coffea"
    try:
        aprocessor = NanoEventsProcessor()
        save(aprocessor, filename)
        newprocessor = load(filename)
        assert "pt" in newprocessor.accumulator
        assert newprocessor.accumulator["pt"].axes == aprocessor.accumulator["pt"].axes
    finally:
        if os.path.exists(filename):
            os.remove(filename)
