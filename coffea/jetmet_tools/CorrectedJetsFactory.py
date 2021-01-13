from coffea.jetmet_tools.JECStack import JECStack
import awkward
import numpy
import warnings
from copy import copy
from functools import partial
import operator


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
def rewrap_recordarray(layout, depth, data):
    if isinstance(layout, awkward.layout.RecordArray):
        return lambda: data
    return None


def awkward_rewrap(arr, like_what, gfunc):
    behavior = awkward._util.behaviorof(like_what)
    func = partial(gfunc, data=arr.layout)
    layout = awkward.operations.convert.to_layout(like_what)
    newlayout = awkward._util.recursively_apply(layout, func)
    return awkward._util.wrap(newlayout, behavior=behavior)


def rand_gauss(item):
    def getfunction(layout, depth):
        if isinstance(layout, awkward.layout.NumpyArray) or not isinstance(
            layout, (awkward.layout.Content, awkward.partition.PartitionedArray)
        ):
            return lambda: awkward.layout.NumpyArray(
                numpy.random.normal(size=len(layout)).astype(numpy.float32)
            )
        return None

    out = awkward._util.recursively_apply(
        awkward.operations.convert.to_layout(item), getfunction
    )
    assert out is not None
    return awkward._util.wrap(out, awkward._util.behaviorof(item))


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
    doHybrid = pt_gen > 0

    detSmear = 1 + (jersf - 1) * (jetPt - pt_gen) / jetPt  # because of awkward1.0#367
    stochSmear = 1 + numpy.sqrt(numpy.maximum(jersf ** 2 - 1, 0)) * jersmear

    min_jet_pt = _MIN_JET_ENERGY / numpy.cosh(etaJet)
    min_jet_pt_corr = min_jet_pt / jetPt

    smearfact = awkward.where(doHybrid, detSmear, stochSmear)
    smearfact = awkward.where(
        (smearfact * jetPt) < min_jet_pt, min_jet_pt_corr, smearfact
    )

    def getfunction(layout, depth):
        if isinstance(layout, awkward.layout.NumpyArray) or not isinstance(
            layout, (awkward.layout.Content, awkward.partition.PartitionedArray)
        ):
            return lambda: awkward.layout.NumpyArray(smearfact)
        return None

    smearfact = awkward._util.recursively_apply(
        awkward.operations.convert.to_layout(jetPt), getfunction
    )
    smearfact = awkward._util.wrap(smearfact, awkward._util.behaviorof(jetPt))
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

    def build(self, jets, lazy_cache):
        if lazy_cache is None:
            raise Exception(
                "CorrectedJetsFactory requires a awkward-array cache to function correctly."
            )
        if not isinstance(jets, awkward.highlevel.Array):
            raise Exception("'jets' must be an awkward > 1.0.0 array of some kind!")
        fields = awkward.fields(jets)
        if len(fields) == 0:
            raise Exception(
                "Detected awkward: 'jets' must have attributes specified by keys!"
            )
        out = awkward.flatten(jets)
        wrap = partial(awkward_rewrap, like_what=jets, gfunc=rewrap_recordarray)
        scalar_form = awkward.without_parameters(
            out[self.name_map["ptRaw"]]
        ).layout.form

        if len(fields) == 0:
            raise Exception(
                "Empty record, please pass a jet object with at least {self.real_sig} defined!"
            )

        # take care of nominal JEC (no JER if available)
        out[self.name_map["JetPt"] + "_orig"] = out[self.name_map["JetPt"]]
        out[self.name_map["JetMass"] + "_orig"] = out[self.name_map["JetMass"]]
        if self.treat_pt_as_raw:
            out[self.name_map["ptRaw"]] = out[self.name_map["JetPt"]]
            out[self.name_map["massRaw"]] = out[self.name_map["JetMass"]]

        jec_name_map = dict(self.name_map)
        jec_name_map["JetPt"] = jec_name_map["ptRaw"]
        jec_name_map["JetMass"] = jec_name_map["massRaw"]
        if self.jec_stack.jec is not None:
            jec_args = {k: out[jec_name_map[k]] for k in self.jec_stack.jec.signature}
            out["jet_energy_correction"] = self.jec_stack.jec.getCorrection(
                **jec_args, form=scalar_form, lazy_cache=lazy_cache
            )
        else:
            out["jet_energy_correction"] = awkward.without_parameters(
                awkward.ones_like(out[self.name_map["JetPt"]])
            )

        # finally the lazy binding to the JEC
        init_pt = partial(
            awkward.virtual,
            operator.mul,
            args=(out["jet_energy_correction"], out[self.name_map["ptRaw"]]),
            cache=lazy_cache,
        )
        init_mass = partial(
            awkward.virtual,
            operator.mul,
            args=(out["jet_energy_correction"], out[self.name_map["massRaw"]]),
            cache=lazy_cache,
        )

        out[self.name_map["JetPt"]] = init_pt(length=len(out), form=scalar_form)
        out[self.name_map["JetMass"]] = init_mass(length=len(out), form=scalar_form)

        out[self.name_map["JetPt"] + "_jec"] = init_pt(
            length=len(out), form=scalar_form
        )
        out[self.name_map["JetMass"] + "_jec"] = init_mass(
            length=len(out), form=scalar_form
        )

        # in jer we need to have a stash for the intermediate JEC products
        has_jer = False
        if self.jec_stack.jer is not None and self.jec_stack.jersf is not None:
            has_jer = True
            jer_name_map = dict(self.name_map)
            jer_name_map["JetPt"] = jer_name_map["JetPt"] + "_jec"
            jer_name_map["JetMass"] = jer_name_map["JetMass"] + "_jec"

            jerargs = {k: out[jer_name_map[k]] for k in self.jec_stack.jer.signature}
            out["jet_energy_resolution"] = self.jec_stack.jer.getResolution(
                **jerargs, form=scalar_form, lazy_cache=lazy_cache
            )

            jersfargs = {
                k: out[jer_name_map[k]] for k in self.jec_stack.jersf.signature
            }
            out[
                "jet_energy_resolution_scale_factor"
            ] = self.jec_stack.jersf.getScaleFactor(
                **jersfargs, form=_JERSF_FORM, lazy_cache=lazy_cache
            )

            out["jet_resolution_rand_gauss"] = awkward.virtual(
                rand_gauss,
                args=(out[self.name_map["JetPt"] + "_orig"],),
                cache=lazy_cache,
                length=len(out),
                form=scalar_form,
            )

            init_jerc = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    0,
                    self.forceStochastic,
                    out[jer_name_map["ptGenJet"]],
                    out[jer_name_map["JetPt"]],
                    out[jer_name_map["JetEta"]],
                    out["jet_energy_resolution"],
                    out["jet_resolution_rand_gauss"],
                    out["jet_energy_resolution_scale_factor"],
                ),
                cache=lazy_cache,
            )
            out["jet_energy_resolution_correction"] = init_jerc(
                length=len(out), form=scalar_form
            )

            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    out["jet_energy_resolution_correction"],
                    out[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    out["jet_energy_resolution_correction"],
                    out[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )
            out[self.name_map["JetPt"]] = init_pt_jer(length=len(out), form=scalar_form)
            out[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )

            out[self.name_map["JetPt"] + "_jer"] = init_pt_jer(
                length=len(out), form=scalar_form
            )
            out[self.name_map["JetMass"] + "_jer"] = init_mass_jer(
                length=len(out), form=scalar_form
            )

            # JER systematics
            jerc_up = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    1,
                    self.forceStochastic,
                    out[jer_name_map["ptGenJet"]],
                    out[jer_name_map["JetPt"]],
                    out[jer_name_map["JetEta"]],
                    out["jet_energy_resolution"],
                    out["jet_resolution_rand_gauss"],
                    out["jet_energy_resolution_scale_factor"],
                ),
                cache=lazy_cache,
            )
            up = awkward.flatten(jets)
            up["jet_energy_resolution_correction"] = jerc_up(
                length=len(out), form=scalar_form
            )
            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    up["jet_energy_resolution_correction"],
                    out[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    up["jet_energy_resolution_correction"],
                    out[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )
            up[self.name_map["JetPt"]] = init_pt_jer(length=len(out), form=scalar_form)
            up[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )

            jerc_down = partial(
                awkward.virtual,
                jer_smear,
                args=(
                    2,
                    self.forceStochastic,
                    out[jer_name_map["ptGenJet"]],
                    out[jer_name_map["JetPt"]],
                    out[jer_name_map["JetEta"]],
                    out["jet_energy_resolution"],
                    out["jet_resolution_rand_gauss"],
                    out["jet_energy_resolution_scale_factor"],
                ),
                cache=lazy_cache,
            )
            down = awkward.flatten(jets)
            down["jet_energy_resolution_correction"] = jerc_down(
                length=len(out), form=scalar_form
            )
            init_pt_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    down["jet_energy_resolution_correction"],
                    out[jer_name_map["JetPt"]],
                ),
                cache=lazy_cache,
            )
            init_mass_jer = partial(
                awkward.virtual,
                operator.mul,
                args=(
                    down["jet_energy_resolution_correction"],
                    out[jer_name_map["JetMass"]],
                ),
                cache=lazy_cache,
            )
            down[self.name_map["JetPt"]] = init_pt_jer(
                length=len(out), form=scalar_form
            )
            down[self.name_map["JetMass"]] = init_mass_jer(
                length=len(out), form=scalar_form
            )
            out["JER"] = awkward.zip(
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
            juncargs = {k: out[juncnames[k]] for k in self.jec_stack.junc.signature}
            juncs = self.jec_stack.junc.getUncertainty(**juncargs)
            for name, func in juncs:
                out[f"jet_energy_uncertainty_{name}"] = func

                def junc_smeared_val(uncvals, up_down, variable):
                    return uncvals[:, up_down] * variable

                up = awkward.flatten(jets)
                up[self.name_map["JetPt"]] = awkward.virtual(
                    junc_smeared_val,
                    args=(
                        out[f"jet_energy_uncertainty_{name}"],
                        0,
                        out[juncnames["JetPt"]],
                    ),
                    length=len(out),
                    form=scalar_form,
                    cache=lazy_cache,
                )
                up[self.name_map["JetMass"]] = awkward.virtual(
                    junc_smeared_val,
                    args=(
                        out[f"jet_energy_uncertainty_{name}"],
                        0,
                        out[juncnames["JetMass"]],
                    ),
                    length=len(out),
                    form=scalar_form,
                    cache=lazy_cache,
                )

                down = awkward.flatten(jets)
                down[self.name_map["JetPt"]] = awkward.virtual(
                    junc_smeared_val,
                    args=(
                        out[f"jet_energy_uncertainty_{name}"],
                        1,
                        out[juncnames["JetPt"]],
                    ),
                    length=len(out),
                    form=scalar_form,
                    cache=lazy_cache,
                )
                down[self.name_map["JetMass"]] = awkward.virtual(
                    junc_smeared_val,
                    args=(
                        out[f"jet_energy_uncertainty_{name}"],
                        1,
                        out[juncnames["JetMass"]],
                    ),
                    length=len(out),
                    form=scalar_form,
                    cache=lazy_cache,
                )
                out[f"JES_{name}"] = awkward.zip(
                    {"up": up, "down": down}, depth_limit=1, with_name="JetSystematic"
                )

        return wrap(out)
