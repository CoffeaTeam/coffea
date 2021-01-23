from coffea.jetmet_tools.JECStack import JECStack
import awkward
import numpy
import warnings
from copy import copy


class CorrectedMETFactory(object):
    def __init__(self, name_map):
        for name in [
            "METpt",
            "METphi",
            "JetPt",
            "JetPhi",
            "ptRaw",
            "UnClusteredEnergyDeltaX",
            "UnClusteredEnergyDeltaY",
        ]:
            if name not in name_map or name_map[name] is None:
                raise ValueError(
                    f"There is no name mapping for {name}, which is needed for CorrectedMETFactory"
                )

        self.name_map = name_map

    def build(self, MET, corrected_jets, lazy_cache):
        if lazy_cache is None:
            raise Exception(
                "CorrectedMETFactory requires a awkward-array cache to function correctly."
            )
        if not isinstance(MET, awkward.highlevel.Array) or not isinstance(
            corrected_jets, awkward.highlevel.Array
        ):
            raise Exception(
                "'MET' and 'corrected_jets' must be an awkward array of some kind!"
            )

        length = len(MET)
        form = awkward.forms.RecordForm(
            {
                "pt": MET[self.name_map["METpt"]].layout.form,
                "phi": MET[self.name_map["METphi"]].layout.form,
            },
        )
        out = copy(MET)
        out[self.name_map["METpt"] + "_orig"] = MET[self.name_map["METpt"]]
        out[self.name_map["METphi"] + "_orig"] = MET[self.name_map["METphi"]]

        def corrected_polar_met(
            met_pt, met_phi, jet_pt, jet_phi, jet_pt_orig, deltas=None
        ):
            sj, cj = numpy.sin(jet_phi), numpy.cos(jet_phi)
            x = met_pt * numpy.cos(met_phi) + awkward.sum(
                jet_pt * cj - jet_pt_orig * cj, axis=1
            )
            y = met_pt * numpy.sin(met_phi) + awkward.sum(
                jet_pt * sj - jet_pt_orig * sj, axis=1
            )
            if deltas:
                positive, dx, dy = deltas
                x = x + dx if positive else x - dx
                y = y + dy if positive else y - dy
            return awkward.zip({"pt": numpy.hypot(x, y), "phi": numpy.arctan2(y, x)})

        corrected_met = awkward.virtual(
            corrected_polar_met,
            args=(
                MET[self.name_map["METpt"]],
                MET[self.name_map["METphi"]],
                corrected_jets[self.name_map["JetPt"]],
                corrected_jets[self.name_map["JetPhi"]],
                corrected_jets[self.name_map["ptRaw"]],
            ),
            length=length,
            form=form,
            cache=lazy_cache,
        )
        out[self.name_map["METpt"]] = awkward.virtual(
            lambda: corrected_met["pt"],
            length=length,
            form=form.contents["pt"],
            cache=lazy_cache,
        )
        out[self.name_map["METphi"]] = awkward.virtual(
            lambda: corrected_met["phi"],
            length=length,
            form=form.contents["phi"],
            cache=lazy_cache,
        )

        def make_unclustered_variant(positive, deltaX, deltaY):
            corrected_met = awkward.virtual(
                corrected_polar_met,
                args=(
                    MET[self.name_map["METpt"]],
                    MET[self.name_map["METphi"]],
                    corrected_jets[self.name_map["JetPt"]],
                    corrected_jets[self.name_map["JetPhi"]],
                    corrected_jets[self.name_map["ptRaw"]],
                    (positive, deltaX, deltaY),
                ),
                length=length,
                form=form,
                cache=lazy_cache,
            )
            variant = copy(MET)
            variant[self.name_map["METpt"]] = awkward.virtual(
                lambda: corrected_met["pt"],
                length=length,
                form=form.contents["pt"],
                cache=lazy_cache,
            )
            variant[self.name_map["METphi"]] = awkward.virtual(
                lambda: corrected_met["phi"],
                length=length,
                form=form.contents["phi"],
                cache=lazy_cache,
            )
            return variant

        unclus_up = make_unclustered_variant(
            True,
            MET[self.name_map["UnClusteredEnergyDeltaX"]],
            MET[self.name_map["UnClusteredEnergyDeltaY"]],
        )
        unclus_down = make_unclustered_variant(
            False,
            MET[self.name_map["UnClusteredEnergyDeltaX"]],
            MET[self.name_map["UnClusteredEnergyDeltaY"]],
        )
        out["MET_UnclusteredEnergy"] = awkward.zip(
            {"up": unclus_up, "down": unclus_down},
            depth_limit=1,
            with_name="METSystematic",
        )

        def make_jet_variant(name, jet_collection):
            corrected_met = awkward.virtual(
                corrected_polar_met,
                args=(
                    MET[self.name_map["METpt"]],
                    MET[self.name_map["METphi"]],
                    jet_collection[self.name_map["JetPt"]],
                    jet_collection[self.name_map["JetPhi"]],
                    jet_collection[self.name_map["ptRaw"]],
                ),
                length=length,
                form=form,
                cache=lazy_cache,
            )
            variant = copy(MET)
            variant[self.name_map["METpt"]] = awkward.virtual(
                lambda: corrected_met["pt"],
                length=length,
                form=form.contents["pt"],
                cache=lazy_cache,
            )
            variant[self.name_map["METphi"]] = awkward.virtual(
                lambda: corrected_met["phi"],
                length=length,
                form=form.contents["phi"],
                cache=lazy_cache,
            )
            return variant

        for unc in filter(
            lambda x: x.startswith(("JER", "JES")), awkward.fields(corrected_jets)
        ):
            up = make_jet_variant(unc, corrected_jets[unc].up)
            down = make_jet_variant(unc, corrected_jets[unc].down)
            out[unc] = awkward.zip(
                {"up": up, "down": down}, depth_limit=1, with_name="METSystematic"
            )
        return out

    def uncertainties(self):
        return ["MET_UnclusteredEnergy"]
