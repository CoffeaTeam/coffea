import awkward
from copy import copy
from coffea.nanoevents.methods.base import behavior, Systematic


@awkward.mixin_class(behavior)
class UpDownSystematic(Systematic):
    """An example instance of a simple systematic with only up/down variations."""

    _udmap = {"up": 0, "down": 1}

    def _build_variations(self, name, what, varying_function, *args, **kwargs):
        whatarray = self[what]

        self["__systematics__", f"__{name}__"] = awkward.virtual(
            varying_function,
            args=(whatarray, *args),
            kwargs=kwargs,
            length=len(whatarray),
        )

    def describe_variations(self):
        return "Muon", ["up", "down"]

    def get_variation(self, name, what, updown):
        fields = awkward.fields(self)
        fields.remove("__systematics__")

        the_type, variations = self.describe_variations()

        varied = self["__systematics__", f"__{name}__", :, self._udmap[updown]]

        params = copy(self.layout.parameters)
        params["variation"] = f"{name}-{what}-{updown}"

        return awkward.zip(
            {field: self[field] if field != what else varied for field in fields},
            depth_limit=1,
            parameters=params,
            behavior=self.behavior,
            with_name=the_type,
        )

    def up(self, name, what):
        return awkward.virtual(
            self.get_variation,
            args=(name, what, "up"),
            length=len(self),
            parameters=self[what].layout.parameters,
        )

    def down(self, name, what):
        return awkward.virtual(
            self.get_variation,
            args=(name, what, "down"),
            length=len(self),
            parameters=self[what].layout.parameters,
        )


behavior[("__typestr__", "UpDownSystematic")] = "UpDownSystematic"
Systematic.add_kind("UpDownSystematic")
