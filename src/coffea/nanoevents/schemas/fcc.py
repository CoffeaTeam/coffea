import copy
import re

from coffea.nanoevents import transforms
from coffea.nanoevents.methods import vector
from coffea.nanoevents.schemas.base import BaseSchema, zip_forms
from coffea.nanoevents.util import concat

# Collection Regex #
# Any branch name with a forward slash '/'
# Example: 'ReconstructedParticles/ReconstructedParticles.energy'
_all_collections = re.compile(r".*[\/]+.*")

# Any branch name with a trailing underscore and an integer n between 0 to 9
# Example: 'EFlowPhoton_1'
_trailing_under = re.compile(r".*_[0-9]")

# Any branch name with a hashtag '#'
# Example: 'ReconstructedParticles#0/ReconstructedParticles#0.index'
_idxs = re.compile(r".*[\#]+.*")

# Any branch name with '[' and ']'
# Example: 'ReconstructedParticles/ReconstructedParticles.covMatrix[10]'
_square_braces = re.compile(r".*\[.*\]")


__dask_capable__ = True


def sort_dict(d):
    """Sort a dictionary by key"""
    return {k: d[k] for k in sorted(d)}


class FCCSchema(BaseSchema):
    """
    Schema-builder for Future Circular Collider pregenerated samples.
    https://fcc-physics-events.web.cern.ch/

    This version is tested on the Spring2021 p8_ee_ZH_ecm240 sample
    https://fcc-physics-events.web.cern.ch/FCCee/delphes/spring2021/idea/
    /eos/experiment/fcc/ee/generation/DelphesEvents/spring2021/IDEA/p8_ee_ZH_ecm240/events_082532938.root
    The FCC samples follow the edm4hep structure.
    https://edm4hep.web.cern.ch/index.html

    FCCSchema inherits from the BaseSchema and returns all the collections as a base.Nanoevents record.
    - Branches with vector components like
            "ReconstructedParticles/ReconstructedParticles.referencePoint.x",
            "ReconstructedParticles/ReconstructedParticles.referencePoint.y" and
            "ReconstructedParticles/ReconstructedParticles.referencePoint.z",
            are zipped together to form the "ReconstructedParticles/ReconstructedParticles.referencePoint" subcollection.
            (see FCCSchema._create_subcollections)
        This is done for all the branches except the momentum.[x,y,z] branches
    - Branches like
            "ReconstructedParticles/ReconstructedParticles.energy",
            "ReconstructedParticles/ReconstructedParticles.charge",
            "ReconstructedParticles/ReconstructedParticles.mass",
            "ReconstructedParticles/ReconstructedParticles.referencePoint"(subcollection containing x,y,z),
            ...
            etc
            are zipped together to form the "ReconstructedParticles" collection.
            (see FCCSchema._main_collections)
        The momentum.[x,y,z] branches along with the energy branch (if available) are used to provide the vector.LorentzVector behavior to the collection.
    - Branches with ObjectIDs(indices to another collection) , example
            "ReconstructedParticles#0/ReconstructedParticles#0.index"
             and
            "ReconstructedParticles#0/ReconstructedParticles#0.collectionID"
            are zipped together to form the ""ReconstructedParticlesidx0" collection.
            (see FCCSchema._idx_collections)
    - Branches with a trailing underscore followed by an integer, example
            "EFlowTrack_1/EFlowTrack_1.location",
            "EFlowTrack_1/EFlowTrack_1.D0",
            "EFlowTrack_1/EFlowTrack_1.phi",
            ...
            etc
            are zipped together to form the "EFlowTrack_1" collection.
            (see FCCSchema._trailing_underscore_collections)
    - Other Unknown, empty, or faulty branches are dealt by FCCSchema._unknown_collections on a case by case basis
    """

    __dask_capable__ = True

    mixins_dictionary = {
        "Electron": "ReconstructedParticle",
        "Muon": "ReconstructedParticle",
        "AllMuon": "ReconstructedParticle",
        "EFlowNeutralHadron": "Cluster",
        "Particle": "MCParticle",
        "Photon": "ReconstructedParticle",
        "ReconstructedParticles": "ReconstructedParticle",
        "EFlowPhoton": "Cluster",
        "MCRecoAssociations": "RecoMCParticleLink",
        "MissingET": "ReconstructedParticle",
        "ParticleIDs": "ParticleID",
        "Jet": "ReconstructedParticle",
        "EFlowTrack": "Track",
        "*idx": "ObjectID",
    }

    _momentum_fields_e = {
        "energy": "E",
        "momentum.x": "px",
        "momentum.y": "py",
        "momentum.z": "pz",
    }
    _replacement = {**_momentum_fields_e}

    _threevec_fields = {
        "position": ["position.x", "position.y", "position.z"],
        "directionError": ["directionError.x", "directionError.y", "directionError.z"],
        "vertex": ["vertex.x", "vertex.y", "vertex.z"],
        "endpoint": ["endpoint.x", "endpoint.y", "endpoint.z"],
        "referencePoint": ["referencePoint.x", "referencePoint.y", "referencePoint.z"],
        "momentumAtEndpoint": [
            "momentumAtEndpoint.x",
            "momentumAtEndpoint.y",
            "momentumAtEndpoint.z",
        ],
        "spin": ["spin.x", "spin.y", "spin.z"],
    }

    # Cross-References : format: {<index branch name> : <target collection name>}
    all_cross_references = {
        "MCRecoAssociations#1.index": "Particle", #MC to Reco connection
        "MCRecoAssociations#0.index": "ReconstructedParticles", #Reco to MC connection
        "Particle#0.index":"Particle", #Parents
        "Particle#1.index":"Particle", #Daughters
        "Muon#0.index":"ReconstructedParticles", #Matched Muons
        "Electron#0.index":"ReconstructedParticles", #Matched Electrons
    }

    def __init__(self, base_form, version="latest"):
        super().__init__(base_form)
        self._form["fields"], self._form["contents"] = self._build_collections(
            self._form["fields"], self._form["contents"]
        )

    def _idx_collections(self, output, branch_forms, all_collections):
        """
        Groups the Hash-Tagged '#' branches into an 'idx' collection (string 'idx' instead of string '#' for a python friendly name)
        - In general, there could be many such idx collections.
        - Each idx collection has two branches --> index and collectionID
        - They define the indexes to another collection (Original type: podio::ObjectID)
        - The ObjectID mixin class is assigned to all idx collections
        - Example:
            "ReconstructedParticles#0/ReconstructedParticles#0.index"
            and
            "ReconstructedParticles#0/ReconstructedParticles#0.collectionID"
            are zipped together to form the "ReconstructedParticlesidx0" collection.
        Note: Since the individual idx collections like ReconstructedParticlesidx0, ReconstructedParticlesidx1, etc don't have the same dimensions,
              I could not zip them together to form a parent branch of the name ReconstructedParticlesidx containing ReconstructedParticlesidx0, ReconstructedParticlesidx1 etc
        """
        field_names = list(branch_forms.keys())

        # Extract all the idx collection names
        # Example: "Jet#0/Jet#0.index" --> "Jet#0"
        idxs = {k.split("/")[0] for k in all_collections if _idxs.match(k)}

        # Remove grouping branches which are generated from BaseSchema and contain no usable info
        # Example: Along with the "Jet#0/Jet#0.index" and "Jet#0/Jet#0.collectionID", BaseSchema may produce "Jet#0" grouping branch.
        # It is an empty branch and needs to be removed
        _grouping_branches = {
            k: branch_forms.pop(k)
            for k in field_names
            if _idxs.match(k) and "/" not in k
        }

        for idx in idxs:
            # Create a Python-friendly name
            # Example: Jet#0 --> Jetidx0
            repl = idx.replace("#", "idx")

            # The content of the collection
            # Example output: {'index':<index form>, 'collectionID':<collectionID form>}
            idx_content = {
                k[2 * len(idx) + 2 :]: branch_forms.pop(k)
                for k in field_names
                if k.startswith(f"{idx}/{idx}.")
            }

            # Zip the index and collectionID and assign the collection name repl; Example: Jetidx0
            output[repl] = zip_forms(
                sort_dict(idx_content),
                idx,
                self.mixins_dictionary.get("*idx", "NanoCollection"),
            )
            output[repl]["content"]["parameters"].update(
                {
                    "collection_name": repl,
                }
            )

        # The Special MCRecoAssociationsidx indexes should be treated differently
        # They have the same dimensions
        # Prepare them to be compatible to later join as 'MCRecoAssociations' collection in the FCCSchema._build_collections function
        # Also see : https://github.com/HEP-FCC/FCCAnalyses/tree/master/examples/basics#association-between-recoparticles-and-montecarloparticles
        if ("MCRecoAssociationsidx0" in output.keys()) and (
            "MCRecoAssociationsidx1" in output.keys()
        ):
            branch_forms["MCRecoAssociations/MCRecoAssociations.reco"] = output.pop(
                "MCRecoAssociationsidx0"
            )
            branch_forms["MCRecoAssociations/MCRecoAssociations.mc"] = output.pop(
                "MCRecoAssociationsidx1"
            )

        return output, branch_forms

    def _main_collections(self, output, branch_forms, all_collections):
        """
        Creates all the regular branches. Regular branches have
        no hash-tag '#' or underscore '_' or braces '[' or ']'.
        Example:
            "ReconstructedParticles/ReconstructedParticles.energy",
            "ReconstructedParticles/ReconstructedParticles.charge",
            "ReconstructedParticles/ReconstructedParticles.mass",
            "ReconstructedParticles/ReconstructedParticles.referencePoint"(subcollection containing x,y,z),
            ...
            etc
            are zipped together to form the "ReconstructedParticles" collection.
        The momentum.[x,y,z] branches along with the energy branch (if available) are used to
        provide the vector.LorentzVector behavior to the collection.
        """
        field_names = list(branch_forms.keys())

        # Extract the regular collection names
        # Example collections: {'Jet', 'ReconstructedParticles', 'MCRecoAssociations', ...}
        collections = {
            collection_name
            for collection_name in all_collections
            if not _idxs.match(collection_name)
            and not _trailing_under.match(collection_name)
        }

        # Zip the collections
        # Example: 'ReconstructedParticles'
        for name in collections:
            # Get the mixin class for the collection, if available, otherwise "NanoCollection" by default
            mixin = self.mixins_dictionary.get(name, "NanoCollection")

            # Content to be zipped together
            # Example collection_content: {'type':<type form>, 'energy':<energy form>, 'momentum.x':<momentum.x form> ...}
            collection_content = {
                k[(2 * len(name) + 2) :]: branch_forms.pop(k)
                for k in field_names
                if k.startswith(f"{name}/{name}.")
            }

            # Change the name of momentum fields, to facilitate the vector.LorentzVector behavior
            # 'energy' --> 'E'
            # 'momentum.x' --> 'px'
            # 'momentum.y' --> 'py'
            # 'momentum.z' --> 'pz'
            collection_content = {
                (k.replace(k, self._replacement[k]) if k in self._replacement else k): v
                for k, v in collection_content.items()
            }

            output[name] = zip_forms(sort_dict(collection_content), name, mixin)
            # Update some metadata
            output[name]["content"]["parameters"].update(
                {
                    "collection_name": name,
                }
            )

            # Remove grouping branches which are generated from BaseSchema and contain no usable info
            # Example: Along with the "Jet/Jet.type","Jet/Jet.energy",etc., BaseSchema may produce "Jet" grouping branch.
            # It is an empty branch and needs to be removed
            if name in field_names:
                branch_forms.pop(name)

        return output, branch_forms

    def _trailing_underscore_collections(self, output, branch_forms, all_collections):
        """
        Create collection with branches have a trailing underscore followed by a integer '*_[0-9]'
        Example:
            "EFlowTrack_1/EFlowTrack_1.location",
            "EFlowTrack_1/EFlowTrack_1.D0",
            "EFlowTrack_1/EFlowTrack_1.phi",
            ...
            etc
            are zipped together to form the "EFlowTrack_1" collection
        Note: - I do not understand how these branches are different from other branches except
                for the obvious naming difference.
              - I found most such branches to be empty..., at least in the test root file.
        """
        # Gather all the collection names with trailing underscore followed by an integer
        # Example: EFlowTrack_1, ParticleIDs_0, EFlowPhoton_0, EFlowPhoton_1, etc.
        collections = {
            collection_name
            for collection_name in all_collections
            if _trailing_under.match(collection_name)
        }

        # Collection names that are trailing underscore followed by an integer but do not
        # have any associated branches with '/', signifying that those collection names
        # are actual singleton branches
        singletons = {
            collection_name
            for collection_name in branch_forms.keys()
            if _trailing_under.match(collection_name)
            and not _all_collections.match(collection_name)
        }

        # Zip branches of a collection that are not singletons
        for name in collections:
            mixin = self.mixins_dictionary.get(name, "NanoCollection")

            # Contents to be zipped
            # Example content: {'type':<type form>, 'chi2':<chi2 form>, 'ndf':<ndf form>, ...}
            field_names = list(branch_forms.keys())
            content = {
                k[(2 * len(name) + 2) :]: branch_forms.pop(k)
                for k in field_names
                if k.startswith(f"{name}/{name}.")
            }

            output[name] = zip_forms(sort_dict(content), name, mixin)
            # Update some metadata
            output[name]["content"]["parameters"].update(
                {
                    "collection_name": name,
                }
            )

        # Singleton branches could be assigned directly without zipping
        for name in singletons:
            output[name] = branch_forms.pop(name)

        return output, branch_forms

    def _unknown_collections(self, output, branch_forms, all_collections):
        """
        Process all the unknown, empty or faulty branches that remain
        after creating all the collections.
        Should be called only after creating all the other relevant collections.

        Note: It is not a neat implementation and needs more testing.
        """
        unlisted = copy.deepcopy(branch_forms)
        for name, content in unlisted.items():
            if content["class"] == "ListOffsetArray":
                if content["content"]["class"] == "RecordArray":
                    # Remove empty branches
                    if len(content["content"]["fields"]) == 0:
                        branch_forms.pop(name)
                        continue
                elif content["content"]["class"] == "RecordArray":
                    # Remove empty branches
                    if len(content["contents"]) == 0:
                        continue
            elif content["class"] == "RecordArray":
                # Remove empty branches
                if len(content["contents"]) == 0:
                    continue
                else:
                    # If the branch is not empty, try to make a collection
                    # assuming good behavior of the branch
                    # Note: It's unlike that such a branch exists

                    # Extract the collection name from the branch
                    record_name = name.split("/")[0]

                    # Contents to be zipped
                    contents = {
                        k[2 * len(record_name) + 2 :]: branch_forms.pop(k)
                        for k in unlisted.keys()
                        if k.startswith(record_name + "/")
                    }
                    output[record_name] = zip_forms(
                        sort_dict(contents),
                        record_name,
                        self.mixins_dictionary.get(record_name, "NanoCollection"),
                    )
            # If a branch is non-empty and is one of its kind (i.e. has no other associated branch)
            # call it a singleton and assign it directly to the output
            else:
                output[name] = content

        return output, branch_forms

    def _create_subcollections(self, branch_forms, all_collections):
        """
        Creates 3-vectors,
        zip _begin and _end branches,
        zip colorFlow.a and colorFlow.a branches
        (Does not zip the momentum fields that are required for
        the overall LorentzVector behavior of a collection)
        """
        field_names = list(branch_forms.keys())

        # Replace square braces in a name for a Python-friendly name; Example: covMatrix[n] --> covMatrix_n_
        for name in field_names:
            if _square_braces.match(name):
                new_name = name.replace("[", "_")
                new_name = new_name.replace("]", "_")
                branch_forms[new_name] = branch_forms.pop(name)

        # Zip _begin and _end branches
        # Example: 'Jet/Jet.clusters_begin', 'Jet/Jet.clusters_end' --> 'Jet/Jet.clusters'
        begin_end_collection = set({})
        for fullname in field_names:
            if fullname.endswith("_begin"):
                begin_end_collection.add(fullname.split("_begin")[0])
            elif fullname.endswith("_end"):
                begin_end_collection.add(fullname.split("_end")[0])
        for name in begin_end_collection:
            begin_end_content = {
                k[len(name) + 1 :]: branch_forms.pop(k)
                for k in field_names
                if k.startswith(name)
            }
            offset_form = {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "i",
                "primitive": "int64",
                "form_key": concat(begin_end_content[list(begin_end_content.keys())[0]]["form_key"],"!offsets"),
            }
            begin_end_content_global = {
                k+"G": transforms.local2global_form(begin_end_content[k], offset_form)
                for k in begin_end_content.keys()
            }
            branch_forms[name] = zip_forms(sort_dict({**begin_end_content,**begin_end_content_global}), name)

        # Zip colorFlow.a and colorFlow.b branches
        # Example: 'Particle/Particle.colorFlow.a', 'Particle/Particle.colorFlow.b' --> 'Particle/Particle.colorFlow'
        color_collection = set({})
        for name in field_names:
            if name.endswith("colorFlow.a"):
                color_collection.add(name.split(".a")[0])
            elif name.endswith("colorFlow.b"):
                color_collection.add(name.split(".b")[0])
        for name in color_collection:
            color_content = {
                k[len(name) + 1 :]: branch_forms.pop(k)
                for k in field_names
                if k.startswith(name)
            }
            branch_forms[name] = zip_forms(sort_dict(color_content), name)

        # Create three_vectors
        # Example: 'Jet/Jet.referencePoint.x', 'Jet/Jet.referencePoint.y', 'Jet/Jet.referencePoint.z' --> 'Jet/Jet.referencePoint'
        for name in all_collections:
            for threevec_name, subfields in self._threevec_fields.items():
                if all(
                    f"{name}/{name}.{subfield}" in field_names for subfield in subfields
                ):
                    content = {
                        "x": branch_forms.pop(f"{name}/{name}.{threevec_name}.x"),
                        "y": branch_forms.pop(f"{name}/{name}.{threevec_name}.y"),
                        "z": branch_forms.pop(f"{name}/{name}.{threevec_name}.z"),
                    }
                    branch_forms[f"{name}/{name}.{threevec_name}"] = zip_forms(
                        sort_dict(content), threevec_name, "ThreeVector"
                    )
        return branch_forms

    def _global_indexers(self, branch_forms, all_collections):

        for cross_ref, target in self.all_cross_references.items():
            collection_name, index_name = cross_ref.split(".")

            #pick up the available fields from target collection to get an offset from
            available_fields = [name for name in branch_forms.keys() if name.startswith(f"{target}/{target}.")]
            
            # By default the idxs have different shape at axis=1 in comparison to target
            # So one needs to fill the empty spaces with -1 which could be removed later
            compatible_index = transforms.grow_local_index_to_target_shape_form(
                branch_forms[f"{collection_name}/{collection_name}.{index_name}"],
                branch_forms[available_fields[0]]
            )
            
            offset_form = {
                "class": "NumpyArray",
                "itemsize": 8,
                "format": "i",
                "primitive": "int64",
                "form_key": concat(*[branch_forms[available_fields[0]]["form_key"],"!offsets",]),
            }
            
            replaced_name = collection_name.replace('#', 'idx')
            branch_forms[f"{target}/{target}.{replaced_name}_{index_name}Global"] = transforms.local2global_form(
                compatible_index,
                offset_form
            )
            
        return branch_forms

    def _build_collections(self, field_names, input_contents):
        """
        Builds all the collections with the necessary behaviors defined in the mixins dictionary
        """
        branch_forms = {k: v for k, v in zip(field_names, input_contents)}

        # All collection names
        # Example: ReconstructedParticles
        all_collections = {
            collection_name.split("/")[0]
            for collection_name in field_names
            if _all_collections.match(collection_name)
        }

        output = {}

        # Create subcollections before creating collections
        # Example: Jet.referencePoint.x, Jet.referencePoint.y, Jet.referencePoint.z --> Jet.referencePoint
        branch_forms = self._create_subcollections(branch_forms, all_collections)

        # Create Global Indexers for all cross references
        branch_forms = self._global_indexers(
            branch_forms, all_collections
        )

        # Process the Hash-Tagged '#' branches
        output, branch_forms = self._idx_collections(
            output, branch_forms, all_collections
        )

        # Process the trailing underscore followed by an integer branches '*_[0-9]'
        output, branch_forms = self._trailing_underscore_collections(
            output, branch_forms, all_collections
        )

        # Process all the other regular branches
        output, branch_forms = self._main_collections(
            output, branch_forms, all_collections
        )

        # Process all the other unknown/faulty/empty branches
        output, branch_forms = self._unknown_collections(
            output, branch_forms, all_collections
        )

        # sort the output by key
        output = sort_dict(output)

        return output.keys(), output.values()

    @classmethod
    def behavior(cls):
        """Behaviors necessary to implement this schema"""
        from coffea.nanoevents.methods import base, fcc

        behavior = {}
        behavior.update(base.behavior)
        behavior.update(vector.behavior)
        behavior.update(fcc.behavior)
        return behavior


class FCC:
    """
    Class to choose the required variant of FCCSchema
    Example: from coffea.nanoevents import FCC
             FCC.get_schema(version='latest')

    Note: For now, only one variant is available, called the latest version, that points
          to the fcc.FCCSchema class. This schema has been made keeping the Spring2021 pre-generated samples.
          More Schema variants could be added later.
    """

    def __init__(self, version="latest"):
        self._version = version

    @classmethod
    def get_schema(cls, version="latest"):
        if version == "latest":
            return FCCSchema
        else:
            pass
