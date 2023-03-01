import awkward
import numpy
import warnings
from functools import partial


_stack_parts = ["jec", "junc", "jer", "jersf"]
_MIN_JET_ENERGY = numpy.array(1e-2, dtype=numpy.float32)
_ONE_F32 = numpy.array(1.0, dtype=numpy.float32)
_ZERO_F32 = numpy.array(0.0, dtype=numpy.float32)
_JERSF_FORM = {
    "class": "NumpyArray",
    "inner_shape": [3],
    "itemsize": 4,
    "format": "f",
    "primitive": "float32",
}


# we're gonna assume that the first record array we encounter is the flattened data
def rewrap_recordarray(layout, depth, data, **kwargs):
    if isinstance(layout, awkward.contents.RecordArray):
        return data
    return None


def awkward_rewrap(arr, like_what, gfunc):
    func = partial(gfunc, data=arr.layout)
    return awkward.transform(func, like_what, behavior=like_what.behavior)


def rand_gauss(item, randomstate):
    def getfunction(layout, depth, **kwargs):
        if isinstance(layout, awkward.contents.NumpyArray) or not isinstance(
            layout, (awkward.contents.Content, awkward.partition.PartitionedArray)
        ):
            return awkward.contents.NumpyArray(
                randomstate.normal(size=len(layout)).astype(numpy.float32)
            )
        return None

    out = awkward.transform(
        getfunction,
        item,
        behavior=item.behavior,
    )
    assert out is not None
    return out


def jer_smear(
    variation,
    forceStochastic,
    pt_gen,
    jetPt,
    etaJet,
    jet_energy_resolution,
    jet_resolution_rand_gauss,
    jet_energy_resolution_scale_factor,
):
    pt_gen = pt_gen if not forceStochastic else None

    if not isinstance(jetPt, awkward.highlevel.Array):
        raise Exception("'jetPt' must be an awkward array of some kind!")

    if forceStochastic:
        pt_gen = awkward.without_parameters(awkward.zeros_like(jetPt))

    jersmear = jet_energy_resolution * jet_resolution_rand_gauss
    jersf = jet_energy_resolution_scale_factor[:, variation]
    deltaPtRel = (jetPt - pt_gen) / jetPt
    doHybrid = (pt_gen > 0) & (numpy.abs(deltaPtRel) < 3 * jet_energy_resolution)

    detSmear = 1 + (jersf - 1) * deltaPtRel
    stochSmear = 1 + numpy.sqrt(numpy.maximum(jersf**2 - 1, 0)) * jersmear

    min_jet_pt = _MIN_JET_ENERGY / numpy.cosh(etaJet)
    min_jet_pt_corr = min_jet_pt / jetPt

    smearfact = awkward.where(doHybrid, detSmear, stochSmear)
    smearfact = awkward.where(
        (smearfact * jetPt) < min_jet_pt, min_jet_pt_corr, smearfact
    )

    def getfunction(layout, depth, **kwargs):
        if isinstance(layout, awkward.contents.NumpyArray) or not isinstance(
            layout, awkward.contents.Content
        ):
            return awkward.contents.NumpyArray(smearfact)
        return None

    smearfact = awkward.transform(getfunction, jetPt, behavior=jetPt.behavior)

    return smearfact


class CorrectedJetsFactory(object):
    def __init__(self, name_map, jec_stack):
        # from PhysicsTools/PatUtils/interface/SmearedJetProducerT.h#L283
        self.forceStochastic = False

        if "ptRaw" not in name_map or name_map["ptRaw"] is None:
            warnings.warn(
                "There is no name mapping for ptRaw,"
                " CorrectedJets will assume that <object>.pt is raw pt!"
            )
            name_map["ptRaw"] = name_map["JetPt"] + "_raw"
        self.treat_pt_as_raw = "ptRaw" not in name_map

        if "massRaw" not in name_map or name_map["massRaw"] is None:
            warnings.warn(
                "There is no name mapping for massRaw,"
                " CorrectedJets will assume that <object>.mass is raw pt!"
            )
            name_map["ptRaw"] = name_map["JetMass"] + "_raw"

        total_signature = set()
        for part in _stack_parts:
            attr = getattr(jec_stack, part)
            if attr is not None:
                total_signature.update(attr.signature)

        missing = total_signature - set(name_map.keys())
        if len(missing) > 0:
            raise Exception(
                f"Missing mapping of {missing} in name_map!"
                + " Cannot evaluate jet corrections!"
                + " Please supply mappings for these variables!"
            )

        if "ptGenJet" not in name_map:
            warnings.warn(
                'Input JaggedCandidateArray must have "ptGenJet" in order to apply hybrid JER smearing method. Stochastic smearing will be applied.'
            )
            self.forceStochastic = True

        self.real_sig = [v for k, v in name_map.items()]
        self.name_map = name_map
        self.jec_stack = jec_stack

    def uncertainties(self):
        out = ["JER"] if self.jec_stack.jer is not None else []
        if self.jec_stack.junc is not None:
            out.extend(["JES_{0}".format(unc) for unc in self.jec_stack.junc.levels])
        return out

    def build(self, jets):
        if not isinstance(jets, awkward.highlevel.Array):
            raise Exception("'jets' must be an awkward > 1.0.0 array of some kind!")
        fields = awkward.fields(jets)
        if len(fields) == 0:
            raise Exception(
                "Empty record, please pass a jet object with at least {self.real_sig} defined!"
            )
        out = awkward.flatten(jets)
        wrap = partial(awkward_rewrap, like_what=jets, gfunc=rewrap_recordarray)
        scalar_form = awkward.without_parameters(
            out[self.name_map["ptRaw"]]
        ).layout.form

        in_dict = {field: out[field] for field in fields}
        out_dict = dict(in_dict)

        # take care of nominal JEC (no JER if available)
        out_dict[self.name_map["JetPt"] + "_orig"] = out_dict[self.name_map["JetPt"]]
        out_dict[self.name_map["JetMass"] + "_orig"] = out_dict[
            self.name_map["JetMass"]
        ]
        if self.treat_pt_as_raw:
            out_dict[self.name_map["ptRaw"]] = out_dict[self.name_map["JetPt"]]
            out_dict[self.name_map["massRaw"]] = out_dict[self.name_map["JetMass"]]

        jec_name_map = dict(self.name_map)
        jec_name_map["JetPt"] = jec_name_map["ptRaw"]
        jec_name_map["JetMass"] = jec_name_map["massRaw"]
        if self.jec_stack.jec is not None:
            jec_args = {
                k: out_dict[jec_name_map[k]] for k in self.jec_stack.jec.signature
            }
            out_dict["jet_energy_correction"] = self.jec_stack.jec.getCorrection(
                **jec_args, form=scalar_form
            )
        else:
            out_dict["jet_energy_correction"] = awkward.without_parameters(
                awkward.ones_like(out_dict[self.name_map["JetPt"]])
            )

        # finally the lazy binding to the JEC
        init_pt = out_dict["jet_energy_correction"] * out_dict[self.name_map["ptRaw"]]
        init_mass = (
            out_dict["jet_energy_correction"] * out_dict[self.name_map["massRaw"]]
        )

        out_dict[self.name_map["JetPt"]] = init_pt
        out_dict[self.name_map["JetMass"]] = init_mass

        out_dict[self.name_map["JetPt"] + "_jec"] = out_dict[self.name_map["JetPt"]]
        out_dict[self.name_map["JetMass"] + "_jec"] = out_dict[self.name_map["JetMass"]]

        # in jer we need to have a stash for the intermediate JEC products
        has_jer = False
        if self.jec_stack.jer is not None and self.jec_stack.jersf is not None:
            has_jer = True
            jer_name_map = dict(self.name_map)
            jer_name_map["JetPt"] = jer_name_map["JetPt"] + "_jec"
            jer_name_map["JetMass"] = jer_name_map["JetMass"] + "_jec"

            jerargs = {
                k: out_dict[jer_name_map[k]] for k in self.jec_stack.jer.signature
            }
            out_dict["jet_energy_resolution"] = self.jec_stack.jer.getResolution(
                **jerargs, form=scalar_form
            )

            jersfargs = {
                k: out_dict[jer_name_map[k]] for k in self.jec_stack.jersf.signature
            }
            out_dict[
                "jet_energy_resolution_scale_factor"
            ] = self.jec_stack.jersf.getScaleFactor(**jersfargs, form=_JERSF_FORM)

            seeds = numpy.array(out_dict[self.name_map["JetPt"] + "_orig"])[
                [0, -1]
            ].view("i4")
            out_dict["jet_resolution_rand_gauss"] = rand_gauss(
                out_dict[self.name_map["JetPt"] + "_orig"],
                numpy.random.Generator(numpy.random.PCG64(seeds)),
            )

            init_jerc = jer_smear(
                0,
                self.forceStochastic,
                out_dict[jer_name_map["ptGenJet"]],
                out_dict[jer_name_map["JetPt"]],
                out_dict[jer_name_map["JetEta"]],
                out_dict["jet_energy_resolution"],
                out_dict["jet_resolution_rand_gauss"],
                out_dict["jet_energy_resolution_scale_factor"],
            )
            out_dict["jet_energy_resolution_correction"] = init_jerc

            init_pt_jer = (
                out_dict["jet_energy_resolution_correction"]
                * out_dict[jer_name_map["JetPt"]]
            )
            init_mass_jer = (
                out_dict["jet_energy_resolution_correction"]
                * out_dict[jer_name_map["JetMass"]]
            )

            out_dict[self.name_map["JetPt"]] = init_pt_jer
            out_dict[self.name_map["JetMass"]] = init_mass_jer

            out_dict[self.name_map["JetPt"] + "_jer"] = out_dict[self.name_map["JetPt"]]
            out_dict[self.name_map["JetMass"] + "_jer"] = out_dict[
                self.name_map["JetMass"]
            ]

            # JER systematics
            jerc_up = jer_smear(
                1,
                self.forceStochastic,
                out_dict[jer_name_map["ptGenJet"]],
                out_dict[jer_name_map["JetPt"]],
                out_dict[jer_name_map["JetEta"]],
                out_dict["jet_energy_resolution"],
                out_dict["jet_resolution_rand_gauss"],
                out_dict["jet_energy_resolution_scale_factor"],
            )
            up = awkward.flatten(jets)
            up["jet_energy_resolution_correction"] = jerc_up

            init_pt_jer = (
                up["jet_energy_resolution_correction"] * out_dict[jer_name_map["JetPt"]]
            )
            init_mass_jer = (
                up["jet_energy_resolution_correction"]
                * out_dict[jer_name_map["JetMass"]],
            )

            up[self.name_map["JetPt"]] = init_pt_jer
            up[self.name_map["JetMass"]] = init_mass_jer

            jerc_down = jer_smear(
                2,
                self.forceStochastic,
                out_dict[jer_name_map["ptGenJet"]],
                out_dict[jer_name_map["JetPt"]],
                out_dict[jer_name_map["JetEta"]],
                out_dict["jet_energy_resolution"],
                out_dict["jet_resolution_rand_gauss"],
                out_dict["jet_energy_resolution_scale_factor"],
            )
            down = awkward.flatten(jets)
            down["jet_energy_resolution_correction"] = jerc_down

            init_pt_jer = (
                down["jet_energy_resolution_correction"]
                * out_dict[jer_name_map["JetPt"]]
            )
            init_mass_jer = (
                down["jet_energy_resolution_correction"]
                * out_dict[jer_name_map["JetMass"]]
            )

            down[self.name_map["JetPt"]] = init_pt_jer
            down[self.name_map["JetMass"]] = init_mass_jer

            out_dict["JER"] = awkward.zip(
                {"up": up, "down": down}, depth_limit=1, with_name="JetSystematic"
            )

        if self.jec_stack.junc is not None:
            juncnames = {}
            juncnames.update(self.name_map)
            if has_jer:
                juncnames["JetPt"] = juncnames["JetPt"] + "_jer"
                juncnames["JetMass"] = juncnames["JetMass"] + "_jer"
            else:
                juncnames["JetPt"] = juncnames["JetPt"] + "_jec"
                juncnames["JetMass"] = juncnames["JetMass"] + "_jec"
            juncargs = {
                k: out_dict[juncnames[k]] for k in self.jec_stack.junc.signature
            }
            juncs = self.jec_stack.junc.getUncertainty(**juncargs)

            def junc_smeared_val(uncvals, up_down, variable):
                return uncvals[:, up_down] * variable

            def build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, updown):
                var_dict = dict(in_dict)
                var_dict[jetpt] = junc_smeared_val(
                    unc,
                    updown,
                    jetpt_orig,
                )
                var_dict[jetmass] = junc_smeared_val(
                    unc,
                    updown,
                    jetmass_orig,
                )
                return awkward.zip(
                    var_dict,
                    depth_limit=1,
                    parameters=out.layout.parameters,
                    behavior=out.behavior,
                )

            def build_variant(unc, jetpt, jetpt_orig, jetmass, jetmass_orig):
                up = build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, 0)
                down = build_variation(unc, jetpt, jetpt_orig, jetmass, jetmass_orig, 1)
                return awkward.zip(
                    {"up": up, "down": down}, depth_limit=1, with_name="JetSystematic"
                )

            for name, func in juncs:
                out_dict[f"jet_energy_uncertainty_{name}"] = func
                out_dict[f"JES_{name}"] = build_variant(
                    func,
                    self.name_map["JetPt"],
                    out_dict[juncnames["JetPt"]],
                    self.name_map["JetMass"],
                    out_dict[juncnames["JetMass"]],
                )

        out_parms = out.layout.parameters
        out_parms["corrected"] = True
        out = awkward.zip(
            out_dict, depth_limit=1, parameters=out_parms, behavior=out.behavior
        )

        return wrap(out)
