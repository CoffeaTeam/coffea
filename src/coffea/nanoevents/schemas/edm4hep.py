import re

from coffea.nanoevents.schemas.base import BaseSchema, nest_jagged_forms, zip_forms

# magic numbers currently used to define cross references
# this will be updated later to make it more general
RECO_PARTICLES = 0
MC_PARTICLES = 1

TRACKSTATE_XREF = 1

_base_collection = re.compile(r".*[\#\/]+.*")
_trailing_under = re.compile(r".*_[0-9]")


class EDM4HEPSchema(BaseSchema):
    """EDM4HEP schema builder

    The EDM4HEP schema is built from all branches found in the supplied file,
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

    __dask_capable__ = True

    # originally this was just _momentum_fields = {"energy", "momentum.x", "momentum.y", "momentum.z"}
    _momentum_fields_e = {"energy", "momentum.x", "momentum.y", "momentum.z"}
    _momentum_fields_m = {"mass", "momentum.x", "momentum.y", "momentum.z"}

    def __init__(self, base_form, *args, **kwargs):
        super().__init__(base_form, *args, **kwargs)
        old_style_form = {
            k: v for k, v in zip(self._form["fields"], self._form["contents"])
        }
        output = self._build_collections(old_style_form)
        self._form["fields"] = [k for k in output.keys()]
        self._form["contents"] = [v for v in output.values()]

    def _build_collections(self, branch_forms):
        # Turn any special classes into the appropriate awkward form
        composite_objects = [
            k
            for k in branch_forms
            if not _base_collection.match(k) and not _trailing_under.match(k)
        ]

        composite_behavior = {  # Dictionary for overriding the default behavior
            "Tracks": "LorentzVector"
        }
        for objname in composite_objects:
            if objname not in [
                "PandoraPFOs",
                "MCParticlesSkimmed",
                "MCTruthRecoLink",
                "RecoMCTruthLink",
                "PandoraClusters",
                "MarlinTrkTracks",
            ]:
                continue
            # grab the * from "objname/objname.*"
            components = {
                k[2 * len(objname) + 2 :]
                for k in branch_forms
                if k.startswith(objname + "/")
            }

            # print(components)

            if all(comp in components for comp in self._momentum_fields_e):
                form = zip_forms(
                    {
                        "x": branch_forms.pop(f"{objname}/{objname}.momentum.x"),
                        "y": branch_forms.pop(f"{objname}/{objname}.momentum.y"),
                        "z": branch_forms.pop(f"{objname}/{objname}.momentum.z"),
                        "t": branch_forms.pop(f"{objname}/{objname}.energy"),
                        "charge": branch_forms.pop(f"{objname}/{objname}.charge"),
                        "pdgId": branch_forms.pop(f"{objname}/{objname}.type"),
                    },
                    objname,
                    composite_behavior.get(objname, "RecoParticle"),
                )
                branch_forms[objname] = form
            elif all(comp in components for comp in self._momentum_fields_m):
                form = zip_forms(
                    {
                        "x": branch_forms.pop(f"{objname}/{objname}.momentum.x"),
                        "y": branch_forms.pop(f"{objname}/{objname}.momentum.y"),
                        "z": branch_forms.pop(f"{objname}/{objname}.momentum.z"),
                        "mass": branch_forms.pop(f"{objname}/{objname}.mass"),
                        "charge": branch_forms.pop(f"{objname}/{objname}.charge"),
                        "pdgId": branch_forms.pop(f"{objname}/{objname}.PDG"),
                    },
                    objname,
                    composite_behavior.get(objname, "MCTruthParticle"),
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
            elif objname == "MCTruthRecoLink" or objname == "RecoMCTruthLink":
                form = zip_forms(
                    {
                        "weight_mc_reco": branch_forms[
                            "MCTruthRecoLink/MCTruthRecoLink.weight"
                        ],
                        "weight_reco_mc": branch_forms[
                            "RecoMCTruthLink/RecoMCTruthLink.weight"
                        ],
                        "reco_index": branch_forms[
                            f"MCTruthRecoLink#{RECO_PARTICLES}/MCTruthRecoLink#{RECO_PARTICLES}.index"
                        ],  # only the weights vary between "MCTruthRecoLink" and "RecoMCTruthLink"
                        "reco_collectionID": branch_forms[
                            f"MCTruthRecoLink#{RECO_PARTICLES}/MCTruthRecoLink#{RECO_PARTICLES}.collectionID"
                        ],
                        "mc_index": branch_forms[
                            f"MCTruthRecoLink#{MC_PARTICLES}/MCTruthRecoLink#{MC_PARTICLES}.index"
                        ],
                        "mc_collectionID": branch_forms[
                            f"MCTruthRecoLink#{MC_PARTICLES}/MCTruthRecoLink#{MC_PARTICLES}.collectionID"
                        ],
                    },
                    objname,
                    composite_behavior.get(objname, "ParticleLink"),
                )
                branch_forms[objname] = form
            elif objname == "PandoraClusters":
                form = zip_forms(
                    {
                        "pt": branch_forms[f"{objname}/{objname}.energy"],  # in mm
                        "theta": branch_forms.pop(f"{objname}/{objname}.iTheta"),
                        "phi": branch_forms.pop(f"{objname}/{objname}.phi"),
                        "energy": branch_forms[f"{objname}/{objname}.energy"],
                    },
                    objname,
                    composite_behavior.get(objname, "Cluster"),
                )
                branch_forms[objname] = form
            elif objname == "MarlinTrkTracks":
                form = zip_forms(
                    {
                        "omega": branch_forms.pop(f"{objname}_{TRACKSTATE_XREF}/{objname}_{TRACKSTATE_XREF}.omega"),
                        "phi": branch_forms.pop(f"{objname}_{TRACKSTATE_XREF}/{objname}_{TRACKSTATE_XREF}.phi"),
                        "tanLambda": branch_forms.pop(f"{objname}_{TRACKSTATE_XREF}/{objname}_{TRACKSTATE_XREF}.tanLambda"),
                        "dEdx": branch_forms.pop(f"{objname}/{objname}.dEdx"),
                    },
                    objname,
                    composite_behavior.get(objname, "Track"),
                )
                branch_forms[objname] = form
            else:
                raise ValueError(
                    f"Unrecognized class with split branches: {components}"
                )

        # Generating collection from branch name
        collections = [
            k for k in branch_forms if k == "PandoraPFOs" or k == "MCParticlesSkimmed"
        ]  # added second case
        collections = {
            "_".join(k.split("_")[:-1])
            for k in collections
            if k.split("_")[-1] != "AK8"
            # Excluding per-event variables with AK8 variants like Mjj and MT
        }

        subcollections = []

        for cname in collections:
            items = sorted(k for k in branch_forms if k.startswith(cname + "_"))
            if len(items) == 0:
                continue

            # Special pattern parsing for <collection>_<subcollection>Counts branches
            countitems = [x for x in items if x.endswith("Counts")]
            subcols = {x[:-6] for x in countitems}  # List of subcollection names
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
                    print(collection["class"])
                    raise NotImplementedError(
                        f"{cname} isn't a jagged array, not sure what to do"
                    )
                for item in items:
                    Itemname = item[len(cname) + 1 :]
                    collection["content"]["fields"].append(Itemname)
                    collection["content"]["contents"].append(
                        branch_forms.pop(item)["content"]
                    )

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
        from coffea.nanoevents.methods import base, edm4hep, vector

        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        behavior.update(edm4hep.behavior)
        return behavior
