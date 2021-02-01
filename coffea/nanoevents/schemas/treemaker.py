from coffea.nanoevents import transforms
from coffea.nanoevents.util import quote, concat
from .base import BaseSchema, listarray_form, zip_forms, nest_jagged_forms


class TreeMakerSchema(BaseSchema):
    """TreeMaker schema builder

    The TreeMaker schema is built from all branches found in the supplied file, based on
    the naming pattern of the branches. From those arrays, TreeMaker collections are formed
    as collections of branches grouped by name, where:

    FIX ME

    All collections are then zipped into one `base.NanoEvents` record and returned.
    """

    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        # Turn any special classes into the appropriate awkward form
        composite_objects = list(set(k.split("/")[0] for k in branch_forms if "/" in k))
        for objname in composite_objects:
            # grab the * from "objname/objname.*"
            components = set(
                k[2 * len(objname) + 2 :]
                for k in branch_forms
                if k.startswith(objname + "/")
            )
            if components == {
                "fCoordinates.fPt",
                "fCoordinates.fEta",
                "fCoordinates.fPhi",
                "fCoordinates.fE",
            }:
                form = zip_forms(
                    {
                        "pt": branch_forms.pop(f"{objname}/{objname}.fCoordinates.fPt"),
                        "eta": branch_forms.pop(
                            f"{objname}/{objname}.fCoordinates.fEta"
                        ),
                        "phi": branch_forms.pop(
                            f"{objname}/{objname}.fCoordinates.fPhi"
                        ),
                        "energy": branch_forms.pop(
                            f"{objname}/{objname}.fCoordinates.fE"
                        ),
                    },
                    objname,
                    "PtEtaPhiELorentzVector",
                )
                print("PtEtaPhiELorentzVector:", objname)
                branch_forms[objname] = form
            elif components == {
                "fCoordinates.fX",
                "fCoordinates.fY",
                "fCoordinates.fZ",
            }:
                form = zip_forms(
                    {
                        "x": branch_forms.pop(f"{objname}/{objname}.fCoordinates.fX"),
                        "y": branch_forms.pop(f"{objname}/{objname}.fCoordinates.fY"),
                        "z": branch_forms.pop(f"{objname}/{objname}.fCoordinates.fZ"),
                    },
                    objname,
                    "Point",
                )
                print("Point:", objname)
                branch_forms[objname] = form
            else:
                raise ValueError(
                    f"Unrecognized class with split branches: {components}"
                )

        collections = [
            "Electrons",
            "GenElectrons",
            "GenJets",
            "GenJetsAK8",
            "GenMuons",
            "GenParticles",
            "GenTaus",
            "GenVertices",
            "Jets",
            "JetsAK8",
            "JetsAK8_subjets",
            "JetsAK15",
            "Muons",
            "PrimaryVertices",
            "TAPElectronTracks",
            "TAPMuonTracks",
            "TAPPionTracks",
            "Tracks",
        ]
        for cname in collections:
            items = sorted(k for k in branch_forms if k.startswith(cname + "_"))
            if len(items) == 0:
                continue
            if cname == "JetsAK8":
                items = [k for k in items if not k.startswith("JetsAK8_subjets")]
                items.append("JetsAK8_subjetsCounts")
            if cname == "JetsAK8_subjets":
                items = [k for k in items if not k.endswith("Counts")]
            if cname not in branch_forms:
                collection = zip_forms(
                    {k[len(cname) + 1]: branch_forms.pop(k) for k in items}, cname
                )
                branch_forms[cname] = collection
            else:
                collection = branch_forms[cname]
                if not collection["class"].startswith("ListOffsetArray"):
                    raise NotImplementedError(
                        f"{cname} isn't a jagged array, not sure what to do"
                    )
                for item in items:
                    itemname = item[len(cname) + 1 :]
                    collection["content"]["contents"][itemname] = branch_forms.pop(
                        item
                    )["content"]

        nest_jagged_forms(
            branch_forms["JetsAK8"],
            branch_forms.pop("JetsAK8_subjets"),
            "subjetsCounts",
            "subjets",
        )

        return branch_forms

    @property
    def behavior(self):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import base, vector

        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        return behavior
