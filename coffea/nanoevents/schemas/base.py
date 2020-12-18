from coffea.nanoevents import transforms
from coffea.nanoevents.util import quote, concat


def listarray_form(content, offsets):
    if offsets["class"] != "NumpyArray":
        raise ValueError
    if offsets["primitive"] == "int32":
        arrayclass = "ListOffsetArray32"
        offsetstype = "i32"
    elif offsets["primitive"] == "int64":
        arrayclass = "ListOffsetArray64"
        offsetstype = "i64"
    else:
        raise ValueError("Unrecognized offsets data type")
    return {
        "class": arrayclass,
        "offsets": offsetstype,
        "content": content,
        "form_key": concat(offsets["form_key"], "!skip"),
    }


def zip_forms(forms, name, record_name=None, offsets=None):
    if not isinstance(forms, dict):
        raise ValueError("Expected a dictionary")
    if all(form["class"].startswith("ListOffsetArray") for form in forms.values()):
        first = next(iter(forms.values()))
        if not all(form["class"] == first["class"] for form in forms.values()):
            raise ValueError
        if not all(form["offsets"] == first["offsets"] for form in forms.values()):
            raise ValueError
        record = {
            "class": "RecordArray",
            "contents": {k: form["content"] for k, form in forms.items()},
            "form_key": quote("!invalid," + name),
        }
        if record_name is not None:
            record["parameters"] = {"__record__": record_name}
        if offsets is None:
            return {
                "class": first["class"],
                "offsets": first["offsets"],
                "content": record,
                "form_key": first["form_key"],
            }
        else:
            return listarray_form(record, offsets)
    elif all(form["class"] == "NumpyArray" for form in forms.values()):
        record = {
            "class": "RecordArray",
            "contents": {k: form for k, form in forms.items()},
            "form_key": quote("!invalid," + name),
        }
        if record_name is not None:
            record["parameters"] = {"__record__": record_name}
        return record
    else:
        raise NotImplementedError("Cannot zip forms")


def nest_jagged_forms(parent, child, counts_name, name):
    """Place child listarray inside parent listarray as a double-jagged array"""
    if not parent["class"].startswith("ListOffsetArray"):
        raise ValueError
    if parent["content"]["class"] != "RecordArray":
        raise ValueError
    if not child["class"].startswith("ListOffsetArray"):
        raise ValueError
    counts = parent["content"]["contents"][counts_name]
    offsets = transforms.counts2offsets_form(counts)
    inner = listarray_form(child["content"], offsets)
    parent["content"]["contents"][name] = inner


class BaseSchema:
    """Base schema builder

    The basic schema is essentially unchanged from the original ROOT file.
    A top-level `base.NanoEvents` object is returned, where each original branch
    form is accessible as a direct descendant.
    """

    behavior = {}

    def __init__(self, base_form):
        params = dict(base_form.get("parameters", {}))
        params["__record__"] = "NanoEvents"
        params.setdefault("metadata", {})
        self._form = {
            "class": "RecordArray",
            "contents": base_form["contents"],
            "parameters": params,
            "form_key": "",
        }

    @property
    def form(self):
        """Awkward form of this schema"""
        return self._form

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import base

        return base.behavior
