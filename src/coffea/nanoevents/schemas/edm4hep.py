import re

from coffea.nanoevents import transforms
from coffea.nanoevents.schemas.base import BaseSchema, nest_jagged_forms, zip_forms
from coffea.nanoevents.util import concat

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
                "MCTruthClusterLink",
                "ClusterMCTruthLink",
                "MarlinTrkTracks",
                "MCTruthMarlinTrkTracksLink",
                "MarlinTrkTracksMCTruthLink",
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
                form["content"]["parameters"]["collection_name"] = objname
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
                        "parents_begin": branch_forms.pop(f"{objname}/{objname}.parents_begin"),
                        "parents_end": branch_forms.pop(f"{objname}/{objname}.parents_end"),
                        "daughters_begin": branch_forms.pop(f"{objname}/{objname}.daughters_begin"),
                        "daughters_end": branch_forms.pop(f"{objname}/{objname}.daughters_end"),
                    },
                    objname,
                    composite_behavior.get(objname, "MCTruthParticle"),
                )
                form["content"]["parameters"]["collection_name"] = objname
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
                form["content"]["parameters"]["collection_name"] = objname
                branch_forms[objname] = form
            elif objname == "MCTruthRecoLink" or objname == "RecoMCTruthLink":
                pfos_offsets_src = (
                    branch_forms["PandoraPFOs/PandoraPFOs.momentum.x"]
                    if "PandoraPFOs/PandoraPFOs.momentum.x" in branch_forms
                    else branch_forms["PandoraPFOs"]
                )
                pfos_offsets_form = {
                    "class": "NumpyArray",
                    "itemsize": 8,
                    "format": "i",
                    "primitive": "int64",
                    "form_key": concat(*[pfos_offsets_src["form_key"], "!offsets"]),
                }

                mc_offsets_src = (
                    branch_forms["MCParticlesSkimmed/MCParticlesSkimmed.momentum.x"]
                    if "MCParticlesSkimmed/MCParticlesSkimmed.momentum.x"
                    in branch_forms
                    else branch_forms["MCParticlesSkimmed"]
                )
                mc_offsets_form = {
                    "class": "NumpyArray",
                    "itemsize": 8,
                    "format": "i",
                    "primitive": "int64",
                    "form_key": concat(*[mc_offsets_src["form_key"], "!offsets"]),
                }

                Greco_index_form = transforms.local2global_form(
                    branch_forms[
                        f"_MCTruthRecoLink_rec/_MCTruthRecoLink_rec.index"
                    ],
                    pfos_offsets_form,
                )
                Gmc_index_form = transforms.local2global_form(
                    branch_forms[
                        f"_MCTruthRecoLink_sim/_MCTruthRecoLink_sim.index"
                    ],
                    mc_offsets_form,
                )

                form = zip_forms(
                    {
                        "Greco_index": Greco_index_form,
                        "Gmc_index": Gmc_index_form,
                        "weight_mc_reco": branch_forms[
                            "MCTruthRecoLink/MCTruthRecoLink.weight"
                        ],
                        "weight_reco_mc": branch_forms[
                            "RecoMCTruthLink/RecoMCTruthLink.weight"
                        ],
                        "reco_index": branch_forms[
                            f"_MCTruthRecoLink_rec/_MCTruthRecoLink_rec.index"
                        ],  # only the weights vary between "MCTruthRecoLink" and "RecoMCTruthLink"
                        "reco_collectionID": branch_forms[
                            f"_MCTruthRecoLink_rec/_MCTruthRecoLink_rec.collectionID"
                        ],
                        "mc_index": branch_forms[
                            f"_MCTruthRecoLink_sim/_MCTruthRecoLink_sim.index"
                        ],
                        "mc_collectionID": branch_forms[
                            f"_MCTruthRecoLink_sim/_MCTruthRecoLink_sim.collectionID"
                        ],
                    },
                    objname,
                    composite_behavior.get(objname, "ParticleLink"),
                )
                branch_forms[objname] = form
            elif objname == "MCTruthClusterLink" or objname == "ClusterMCTruthLink":
                cluster_offsets_src = (
                    branch_forms["PandoraClusters/PandoraClusters.energy"]
                    if "PandoraClusters/PandoraClusters.energy" in branch_forms
                    else branch_forms["PandoraClusters"]
                )
                cluster_offsets_form = {
                    "class": "NumpyArray",
                    "itemsize": 8,
                    "format": "i",
                    "primitive": "int64",
                    "form_key": concat(*[cluster_offsets_src["form_key"], "!offsets"]),
                }

                mc_offsets_src = (
                    branch_forms["MCParticlesSkimmed/MCParticlesSkimmed.momentum.x"]
                    if "MCParticlesSkimmed/MCParticlesSkimmed.momentum.x"
                    in branch_forms
                    else branch_forms["MCParticlesSkimmed"]
                )
                mc_offsets_form = {
                    "class": "NumpyArray",
                    "itemsize": 8,
                    "format": "i",
                    "primitive": "int64",
                    "form_key": concat(*[mc_offsets_src["form_key"], "!offsets"]),
                }

                Gcluster_index_form = transforms.local2global_form(
                    branch_forms[
                        f"_MCTruthClusterLink_rec/_MCTruthClusterLink_rec.index"
                    ],
                    cluster_offsets_form,
                )
                Gmc_index_form = transforms.local2global_form(
                    branch_forms[
                        f"_MCTruthClusterLink_sim/_MCTruthClusterLink_sim.index"
                    ],
                    mc_offsets_form,
                )

                form = zip_forms(
                    {
                        "Gcluster_index": Gcluster_index_form,
                        "Gmc_index": Gmc_index_form,
                        "weight_mc_cluster": branch_forms[
                            "MCTruthClusterLink/MCTruthClusterLink.weight"
                        ],
                        "weight_cluster_mc": branch_forms[
                            "ClusterMCTruthLink/ClusterMCTruthLink.weight"
                        ],
                        "cluster_index": branch_forms[
                            f"_MCTruthClusterLink_rec/_MCTruthClusterLink_rec.index"
                        ],
                        "cluster_collectionID": branch_forms[
                            f"_MCTruthClusterLink_rec/_MCTruthClusterLink_rec.collectionID"
                        ],
                        "mc_index": branch_forms[
                            f"_MCTruthClusterLink_sim/_MCTruthClusterLink_sim.index"
                        ],
                        "mc_collectionID": branch_forms[
                            f"_MCTruthClusterLink_sim/_MCTruthClusterLink_sim.collectionID"
                        ],
                    },
                    objname,
                    composite_behavior.get(objname, "ParticleLink"),
                )
                branch_forms[objname] = form
            elif (
                objname == "MCTruthMarlinTrkTracksLink"
                or objname == "MarlinTrkTracksMCTruthLink"
            ):
                trk_offsets_src = (
                    branch_forms[
                        f"_MarlinTrkTracks_trackStates/_MarlinTrkTracks_trackStates.omega"
                    ]
                    if f"_MarlinTrkTracks_trackStates/_MarlinTrkTracks_trackStates.omega"
                    in branch_forms
                    else branch_forms[
                        "MarlinTrkTracks"
                    ]  
                )
                trk_offsets_form = {
                    "class": "NumpyArray",
                    "itemsize": 8,
                    "format": "i",
                    "primitive": "int64",
                    "form_key": concat(*[trk_offsets_src["form_key"], "!offsets"]),
                }

                mc_offsets_src = (
                    branch_forms["MCParticlesSkimmed/MCParticlesSkimmed.momentum.x"]
                    if "MCParticlesSkimmed/MCParticlesSkimmed.momentum.x"
                    in branch_forms
                    else branch_forms["MCParticlesSkimmed"]
                )
                mc_offsets_form = {
                    "class": "NumpyArray",
                    "itemsize": 8,
                    "format": "i",
                    "primitive": "int64",
                    "form_key": concat(*[mc_offsets_src["form_key"], "!offsets"]),
                }

                Gtrk_index_form = transforms.local2global_form(
                    branch_forms[
                        f"_MCTruthMarlinTrkTracksLink_rec/_MCTruthMarlinTrkTracksLink_rec.index"
                    ],
                    trk_offsets_form,
                )
                Gmc_index_form = transforms.local2global_form(
                    branch_forms[
                        f"_MCTruthMarlinTrkTracksLink_sim/_MCTruthMarlinTrkTracksLink_sim.index"
                    ],
                    mc_offsets_form,
                )

                form = zip_forms(
                    {
                        "Gtrk_index": Gtrk_index_form,
                        "Gmc_index": Gmc_index_form,
                        "weight_mc_trk": branch_forms[
                            "MCTruthMarlinTrkTracksLink/MCTruthMarlinTrkTracksLink.weight"
                        ],
                        "weight_trk_mc": branch_forms[
                            "MarlinTrkTracksMCTruthLink/MarlinTrkTracksMCTruthLink.weight"
                        ],
                        "trk_index": branch_forms[
                            f"_MCTruthMarlinTrkTracksLink_rec/_MCTruthMarlinTrkTracksLink_rec.index"
                        ],
                        "trk_collectionID": branch_forms[
                            f"_MCTruthMarlinTrkTracksLink_rec/_MCTruthMarlinTrkTracksLink_rec.collectionID"
                        ],
                        "mc_index": branch_forms[
                            f"_MCTruthMarlinTrkTracksLink_sim/_MCTruthMarlinTrkTracksLink_sim.index"
                        ],
                        "mc_collectionID": branch_forms[
                            f"_MCTruthMarlinTrkTracksLink_sim/_MCTruthMarlinTrkTracksLink_sim.collectionID"
                        ],
                    },
                    objname,
                    composite_behavior.get(objname, "ParticleLink"),
                )
                branch_forms[objname] = form
            elif objname == "PandoraClusters":
                form = zip_forms(
                    {
                        "pt": branch_forms[f"{objname}/{objname}.energy"], 
                        "theta": branch_forms.pop(f"{objname}/{objname}.iTheta"),
                        "phi": branch_forms.pop(f"{objname}/{objname}.phi"),
                        "energy": branch_forms[f"{objname}/{objname}.energy"],
                    },
                    objname,
                    composite_behavior.get(objname, "Cluster"),
                )
                form["content"]["parameters"]["collection_name"] = objname
                branch_forms[objname] = form
            elif objname == "MarlinTrkTracks":
                form = zip_forms(
                    {
                        "omega": branch_forms.pop(
                            f"_MarlinTrkTracks_trackStates/_MarlinTrkTracks_trackStates.omega"
                        ),
                        "phi": branch_forms.pop(
                            f"_MarlinTrkTracks_trackStates/_MarlinTrkTracks_trackStates.phi"
                        ),
                        "tanLambda": branch_forms.pop(
                            f"_MarlinTrkTracks_trackStates/_MarlinTrkTracks_trackStates.tanLambda"
                        ),
                        # "dEdx": branch_forms.pop(f"_MarlinTrkTracks_trackStates/_MarlinTrkTracks_trackStates.dEdx"),
                    },
                    objname,
                    composite_behavior.get(objname, "Track"),
                )
                form["content"]["parameters"]["collection_name"] = objname
                branch_forms[objname] = form
            else:
                raise ValueError(
                    f"Unrecognized class with split branches: {components}"
                )

        # Generating collection from branch name
        collections = [
            k
            for k in branch_forms
            if k
            in [
                "PandoraPFOs",
                "MCParticlesSkimmed",
                "MCTruthRecoLink",
                "RecoMCTruthLink",
                "PandoraClusters",
                "MCTruthClusterLink",
                "ClusterMCTruthLink",
                "MarlinTrkTracks",
            ]
        ]
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
