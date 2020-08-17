import awkward1


behavior = {}


@awkward1.mixin_class(behavior)
class NanoEvents:
    """NanoEvents mixin class

    This mixin class is used as the top-level type for NanoEvents objects.
    """

    @property
    def metadata(self):
        """Arbitrary metadata"""
        return self.layout.purelist_parameter("metadata")


@awkward1.mixin_class(behavior)
class NanoCollection:
    """A NanoEvents collection

    This mixin provides some helper methods useful for creating cross-references
    and other advanced mixin types.
    """

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

    def _content(self):
        """Internal method to get jagged collection content

        This should only be called on the original unsliced collection array.
        Used with global indexes to resolve cross-references"""
        return self._getlistarray().content

    def _events(self):
        """Internal method to get the originally-constructed NanoEvents

        This can be called at any time from any collection, as long as
        the NanoEventsFactory instance exists."""
        return self.behavior["__events_factory__"].events()
