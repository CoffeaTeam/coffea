import awkward
import dask_awkward
import numpy


def corrected_polar_met(met_pt, met_phi, jet_pt, jet_phi, jet_pt_orig, deltas=None):
    sj, cj = numpy.sin(jet_phi), numpy.cos(jet_phi)
    projx = jet_pt * cj - jet_pt_orig * cj
    x = met_pt * numpy.cos(met_phi) + dask_awkward.map_partitions(
        awkward.sum,
        projx,
        axis=1,
        output_divisions=1,
        meta=awkward.flatten(projx._meta, axis=1),
    )
    projy = jet_pt * sj - jet_pt_orig * sj
    y = met_pt * numpy.sin(met_phi) + dask_awkward.map_partitions(
        awkward.sum,
        projy,
        axis=1,
        output_divisions=1,
        meta=awkward.flatten(projy._meta, axis=1),
    )
    if deltas:
        positive, dx, dy = deltas
        x = x + dx if positive else x - dx
        y = y + dy if positive else y - dy
    return dask_awkward.zip(
        {"pt": numpy.hypot(x, y), "phi": numpy.arctan2(y, x)}, depth_limit=1
    )


class CorrectedMETFactory:
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

    def build(self, in_MET, in_corrected_jets):
        if not isinstance(
            in_MET, (awkward.highlevel.Array, dask_awkward.Array)
        ) or not isinstance(
            in_corrected_jets, (awkward.highlevel.Array, dask_awkward.Array)
        ):
            raise Exception(
                "'MET' and 'corrected_jets' must be an (dask_)awkward array of some kind!"
            )

        MET = in_MET
        if isinstance(in_MET, awkward.highlevel.Array):
            MET = dask_awkward.from_awkward(in_MET, 1)

        corrected_jets = in_corrected_jets
        if isinstance(in_corrected_jets, awkward.highlevel.Array):
            corrected_jets = dask_awkward.from_awkward(in_corrected_jets, 1)

        def make_variant(*args):
            variant = MET
            corrected_met = corrected_polar_met(*args)
            variant = dask_awkward.with_field(variant, corrected_met.pt, "METpt")
            variant = dask_awkward.with_field(variant, corrected_met.phi, "METphi")
            return variant

        def lazy_variant(unc, metpt, metphi, jetpt, jetphi, jetptraw):
            return dask_awkward.zip(
                {
                    "up": make_variant(
                        MET[metpt],
                        MET[metphi],
                        corrected_jets[unc].up[jetpt],
                        corrected_jets[unc].up[jetphi],
                        corrected_jets[unc].up[jetptraw],
                    ),
                    "down": make_variant(
                        MET[metpt],
                        MET[metphi],
                        corrected_jets[unc].down[jetpt],
                        corrected_jets[unc].down[jetphi],
                        corrected_jets[unc].down[jetptraw],
                    ),
                },
                depth_limit=1,
                with_name="METSystematic",
            )

        out = make_variant(
            MET[self.name_map["METpt"]],
            MET[self.name_map["METphi"]],
            corrected_jets[self.name_map["JetPt"]],
            corrected_jets[self.name_map["JetPhi"]],
            corrected_jets[self.name_map["ptRaw"]],
        )
        out = dask_awkward.with_field(
            out, MET[self.name_map["METpt"]], self.name_map["METpt"] + "_orig"
        )
        out = dask_awkward.with_field(
            out, MET[self.name_map["METphi"]], self.name_map["METphi"] + "_orig"
        )

        out_dict = {field: out[field] for field in dask_awkward.fields(out)}

        out_dict["MET_UnclusteredEnergy"] = dask_awkward.zip(
            {
                "up": make_variant(
                    MET[self.name_map["METpt"]],
                    MET[self.name_map["METphi"]],
                    corrected_jets[self.name_map["JetPt"]],
                    corrected_jets[self.name_map["JetPhi"]],
                    corrected_jets[self.name_map["ptRaw"]],
                    (
                        True,
                        MET[self.name_map["UnClusteredEnergyDeltaX"]],
                        MET[self.name_map["UnClusteredEnergyDeltaY"]],
                    ),
                ),
                "down": make_variant(
                    MET[self.name_map["METpt"]],
                    MET[self.name_map["METphi"]],
                    corrected_jets[self.name_map["JetPt"]],
                    corrected_jets[self.name_map["JetPhi"]],
                    corrected_jets[self.name_map["ptRaw"]],
                    (
                        False,
                        MET[self.name_map["UnClusteredEnergyDeltaX"]],
                        MET[self.name_map["UnClusteredEnergyDeltaY"]],
                    ),
                ),
            },
            depth_limit=1,
            with_name="METSystematic",
        )

        for unc in filter(
            lambda x: x.startswith(("JER", "JES")), dask_awkward.fields(corrected_jets)
        ):
            out_dict[unc] = lazy_variant(
                unc,
                self.name_map["METpt"],
                self.name_map["METphi"],
                self.name_map["JetPt"],
                self.name_map["JetPhi"],
                self.name_map["ptRaw"],
            )

        out_parms = out._meta.layout.parameters
        out = dask_awkward.zip(
            out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
        )

        return out

    def uncertainties(self):
        return ["MET_UnclusteredEnergy"]
