from coffea.nanoevents.schemas.base import BaseSchema, zip_forms, nest_jagged_forms


class TreeMakerSchema(BaseSchema):
    """TreeMaker schema builder

    The TreeMaker schema is built from all branches found in the supplied file,
    based on the naming pattern of the branches. There are two steps of to the
    generation of array collections:

    - Objects with vector-like quantities (momentum, coordinate points) in the
      TreeMaker ntuples are stored using ROOT PtEtaPhiEVectors and XYZPoint
      classes with maximum TTree splitting. These variable branches are grouped
      into a single collection with the original object name, with the
      corresponding coordinate variables names mapped to the standard variable
      names for coffea.nanoevents.methods.vector behaviors. For example:
      - The "Jets" branch in a TreeMaker Ntuple branch stores 'PtEtaPhiEVector's
        corresponding to the momentum of AK4 jets. The resulting collection after
        this first step would contain the vector variables in the form of
        Jets.pt, Jets.eta, Jets.phi, Jets.energy, and addition vector quantities
        (px) can be accessed via the usual vector behavior methods.
      - The "PrimaryVertices" branch in a TreeMaker Ntuple branch stores
        'XYZPoint's corresponding to the coordinates of the primary vertices, The
        resulting collection after this first step wold contain the coordinate
        variables in the form of PrimaryVertices.x, PrimaryVertices.y,
        PrimaryVertices.z.

    - Extended quantities of physic objects are stored in the format
      <Object>_<variable>, such as "Jets_jecFactor". Such variables will be
      merged into the collection <Object>, so the branch "Jets_jetFactor" will be
      access to in the array format as "Jets.jecFactor". An exception to the

    All collections are then zipped into one `base.NanoEvents` record and
    returned.
    """

    def __init__(self, base_form):
        super().__init__(base_form)
        self._form["contents"] = self._build_collections(self._form["contents"])

    def _build_collections(self, branch_forms):
        # Turn any special classes into the appropriate awkward form
        composite_objects = list(set(k.split("/")[0] for k in branch_forms if "/" in k))

        composite_behavior = {  # Dictionary for overriding the default behavior
            "Tracks": "LorentzVector"
        }
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
                    composite_behavior.get(objname, "PtEtaPhiELorentzVector"),
                )
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
                    composite_behavior.get(objname, "ThreeVector"),
                )
                branch_forms[objname] = form
            else:
                raise ValueError(
                    f"Unrecognized class with split branches: {components}"
                )

        # Generating collection from branch name
        collections = [k for k in branch_forms if "_" in k]
        collections = set(
            [
                "_".join(k.split("_")[:-1])
                for k in collections
                if k.split("_")[-1] != "AK8"
                # Excluding per-event variables with AK8 variants like Mjj and MT
            ]
        )

        subcollections = []

        for cname in collections:
            items = sorted(k for k in branch_forms if k.startswith(cname + "_"))
            if len(items) == 0:
                continue

            # Special pattern parsing for <collection>_<subcollection>Counts branches
            countitems = [x for x in items if x.endswith("Counts")]
            subcols = set(x[:-6] for x in countitems)  # List of subcollection names
            for subcol in subcols:
                items = [
                    k for k in items if not k.startswith(subcol) or k.endswith("Counts")
                ]
                subname = subcol[len(cname) + 1 :]
                subcollections.append(
                    {
                        "colname": cname,
                        "subcol": subcol,
                        "countname": subname + "Counts",
                        "subname": subname,
                    }
                )

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

        for sub in subcollections:
            nest_jagged_forms(
                branch_forms[sub["colname"]],
                branch_forms.pop(sub["subcol"]),
                sub["countname"],
                sub["subname"],
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
