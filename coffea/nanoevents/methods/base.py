from functools import wraps
import numpy
import awkward1
from coffea.nanoevents.methods.mixin import mixin_class, mixin_method
from coffea.nanoevents.factory import NanoEventsFactory


@mixin_class
class NanoEvents:
    """NanoEvents mixin class"""

    @property
    def metadata(self):
        return self.layout.parameter("metadata")


@mixin_class
class NanoCollection:
    def _getlistarray(self):
        """Do some digging to find the initial listarray"""

        def descend(layout, depth):
            islistarray = isinstance(
                layout,
                (awkward1.layout.ListOffsetArray32, awkward1.layout.ListOffsetArray64,),
            )
            if islistarray and layout.content.parameter("collection_name") is not None:
                return lambda: layout

        return awkward1._util.recursively_apply(self.layout, descend)

    def _starts(self):
        """Internal method to get jagged collection starts

        This should only be called on the original unsliced collection array.
        Used to convert local indexes to global indexes"""
        return numpy.asarray(self._getlistarray().starts).astype("i8")

    def _content(self):
        """Internal method to get jagged collection content

        This should only be called on the original unsliced collection array.
        Used with global indexes to resolve cross-references"""
        return self._getlistarray().content

    def _events(self):
        """Internal method to get the originally-constructed NanoEvents

        This can be called at any time from any collection, as long as
        the NanoEventsFactory instance exists."""
        key = self.layout.purelist_parameter("events_key")
        return NanoEventsFactory.get_events(key)


@mixin_class
class NanoColumn:
    """A column from the original file"""

    pass


def runtime_cache(method):
    """Cache a NanoCollection method in the runtime cache

    This decorator can be placed on any method that is a member
    of a class that subclasses NanoCollection. Such methods should
    only be called on the original unsliced collection array."""

    @wraps(method)
    def cachedmethod(self, *args, **kwargs):
        events_key = self.layout.purelist_parameter("events_key")
        collection_name = self.layout.purelist_parameter("collection_name")
        cache = NanoEventsFactory.get_cache(events_key)
        key = "/".join([events_key, "runtime", collection_name, method.__name__])
        try:
            return cache[key]
        except KeyError:
            return method(*args, **kwargs)

    return cachedmethod
