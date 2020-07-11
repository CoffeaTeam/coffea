import numpy
import awkward1
from coffea.nanoevents.methods.mixin import mixin_class, mixin_method
from coffea.nanoevents.factory import NanoEventsFactory


@mixin_class
class NanoEvents:
    pass


@mixin_class
class NanoCollection:
    def _starts(self):
        layout = self.layout
        if isinstance(layout, awkward1.layout.VirtualArray):
            layout = layout.array
        if not isinstance(layout, awkward1.layout.ListOffsetArray32):
            raise RuntimeError("unexpected type in NanoCollection _starts call")
        return numpy.asarray(layout.starts).astype("i8")

    def _events(self):
        key = self.layout.purelist_parameter("events_key")
        out = NanoEventsFactory.get_events(key)
        if len(out) != len(self):
            raise RuntimeError(
                "Parent events for %r does not match shape. Please only mask the first dimension"
                % self
            )
        return out
