from coffea.nanoevents.methods.mixin import mixin_class, mixin_method
from coffea.nanoevents.factory import NanoEventsFactory


@mixin_class
class NanoEvents:
    pass


@mixin_class
class NanoCollection:
    @property
    def __doc__(self):
        return self.layout.purelist_parameter("__doc__")

    def _events(self):
        key = self.layout.purelist_parameter("events_key")
        return NanoEventsFactory.get_events(key)

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.__doc__ = self.layout.purelist_parameter("__doc__")
