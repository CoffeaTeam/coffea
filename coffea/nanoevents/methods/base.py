"""Basic NanoEvents and NanoCollection mixins"""
import awkward


behavior = {}


@awkward.mixin_class(behavior)
class NanoEvents:
    """NanoEvents mixin class

    This mixin class is used as the top-level type for NanoEvents objects.
    """

    @property
    def metadata(self):
        """Arbitrary metadata"""
        return self.layout.purelist_parameter("metadata")


behavior[("__typestr__", "NanoEvents")] = "event"


@awkward.mixin_class(behavior)
class NanoCollection:
    """A NanoEvents collection

    This mixin provides some helper methods useful for creating cross-references
    and other advanced mixin types.
    """

    def _collection_name(self):
        """The name of the collection (i.e. the field under events where it is found)"""
        return self.layout.purelist_parameter("collection_name")

    def _getlistarray(self):
        """Do some digging to find the initial listarray"""

        def descend(layout, depth):
            islistarray = isinstance(
                layout,
                (
                    awkward.layout.ListOffsetArray32,
                    awkward.layout.ListOffsetArray64,
                ),
            )
            if islistarray and layout.content.parameter("collection_name") is not None:
                return lambda: layout

        return awkward._util.recursively_apply(self.layout, descend)

    def _content(self):
        """Internal method to get jagged collection content

        This should only be called on the original unsliced collection array.
        Used with global indexes to resolve cross-references"""
        return self._getlistarray().content

    def _apply_global_index(self, index):
        """Internal method to take from a collection using a flat index

        This is often necessary to be able to still resolve cross-references on
        reduced arrays or single records.
        """
        if isinstance(index, int):
            out = self._content()[index]
            return awkward.Record(out, behavior=self.behavior)

        def flat_take(layout):
            idx = awkward.Array(layout)
            return self._content()[idx.mask[idx >= 0]]

        def descend(layout, depth):
            if layout.purelist_depth == 1:
                return lambda: flat_take(layout)

        (index,) = awkward.broadcast_arrays(index)
        out = awkward._util.recursively_apply(index.layout, descend)
        return awkward.Array(out, behavior=self.behavior)

    def _events(self):
        """Internal method to get the originally-constructed NanoEvents

        This can be called at any time from any collection, as long as
        the NanoEventsFactory instance exists."""
        return self.behavior["__events_factory__"].events()


__all__ = ["NanoCollection", "NanoEvents"]
