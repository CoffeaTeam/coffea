"""Tools of general use for columnar analysis

These helper classes were previously part of ``coffea.processor``
but have been migrated and updated to be compatible with awkward-array 1.0
"""
import awkward
import dask.array
import dask_awkward
import numpy

import coffea.processor
import coffea.util

import hist
import hist.dask

from collections import namedtuple


class WeightStatistics(coffea.processor.AccumulatorABC):
    def __init__(self, sumw=0.0, sumw2=0.0, minw=numpy.inf, maxw=-numpy.inf, n=0):
        self.sumw = sumw
        self.sumw2 = sumw2
        self.minw = minw
        self.maxw = maxw
        self.n = n

    def __repr__(self):
        return f"WeightStatistics(sumw={self.sumw}, sumw2={self.sumw2}, minw={self.minw}, maxw={self.maxw}, n={self.n})"

    def identity(self):
        return WeightStatistics()

    def add(self, other):
        self.sumw += other.sumw
        self.sumw2 += other.sumw2
        self.minw = min(self.minw, other.minw)
        self.maxw = max(self.maxw, other.maxw)
        self.n += other.n


class Weights:
    """Container for event weights and associated systematic shifts

    This container keeps track of correction factors and systematic
    effects that can be encoded as multiplicative modifiers to the event weight.
    All weights are stored in vector form.

    Parameters
    ----------
        size : int | None
            size of the weight arrays to be handled (i.e. the number of events / instances).
            If None then we expect to operate in delayed mode.
        storeIndividual : bool, optional
            store not only the total weight + variations, but also each individual weight.
            Default is false.
    """

    def __init__(self, size, storeIndividual=False):
        self._weight = None if size is None else numpy.ones(size)
        self._weights = {}
        self._modifiers = {}
        self._weightStats = coffea.processor.dict_accumulator()
        self._storeIndividual = storeIndividual

    @property
    def weightStatistics(self):
        return self._weightStats

    def __add_eager(self, name, weight, weightUp, weightDown, shift):
        """Add a new weight with eager calculation"""
        if isinstance(weight, numpy.ma.MaskedArray):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = weight.filled(1.0)
        self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        self.__add_variation(name, weight, weightUp, weightDown, shift)
        self._weightStats[name] = WeightStatistics(
            weight.sum(),
            (weight**2).sum(),
            weight.min(),
            weight.max(),
            weight.size,
        )

    def __add_delayed(self, name, weight, weightUp, weightDown, shift):
        """Add a new weight with delayed calculation"""
        if isinstance(dask_awkward.type(weight), awkward.types.OptionType):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = dask_awkward.fill_none(weight, 1.0)
        if self._weight is None:
            self._weight = weight
        else:
            self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        self.__add_variation(name, weight, weightUp, weightDown, shift)
        if isinstance(self._weightStats, coffea.processor.dict_accumulator):
            self._weightStats = {}
        self._weightStats[name] = {
            "sumw": dask_awkward.to_dask_array(weight).sum(),
            "sumw2": dask_awkward.to_dask_array(weight**2).sum(),
            "minw": dask_awkward.to_dask_array(weight).min(),
            "maxw": dask_awkward.to_dask_array(weight).max(),
        }

    def add(self, name, weight, weightUp=None, weightDown=None, shift=False):
        """Add a new weight

        Adds a named correction to the event weight, and optionally also associated
        systematic uncertainties.

        Parameters
        ----------
            name : str
                name of correction
            weight : numpy.ndarray
                the nominal event weight associated with the correction
            weightUp : numpy.ndarray, optional
                weight with correction uncertainty shifted up (if available)
            weightDown : numpy.ndarray, optional
                weight with correction uncertainty shifted down. If ``weightUp`` is supplied, and
                the correction uncertainty is symmetric, this can be set to None to auto-calculate
                the down shift as ``1 / weightUp``.
            shift : bool, optional
                if True, interpret weightUp and weightDown as a relative difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if name.endswith("Up") or name.endswith("Down"):
            raise ValueError(
                "Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call"
            )
        weight = coffea.util._ensure_flat(weight, allow_missing=True)
        if isinstance(weight, numpy.ndarray) and isinstance(
            self._weight, numpy.ndarray
        ):
            self.__add_eager(name, weight, weightUp, weightDown, shift)
        elif isinstance(weight, dask_awkward.Array) and isinstance(
            self._weight, (dask_awkward.Array, type(None))
        ):
            self.__add_delayed(name, weight, weightUp, weightDown, shift)
        else:
            raise ValueError(
                f"Incompatible weights: self._weight={type(self.weight)}, weight={type(weight)}"
            )

    def __add_multivariation_eager(
        self, name, weight, modifierNames, weightsUp, weightsDown, shift=False
    ):
        """Add a new weight with multiple variations in eager mode"""
        if isinstance(weight, numpy.ma.MaskedArray):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = weight.filled(1.0)
        self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        # Now loop on the variations
        if len(modifierNames) > 0:
            if len(modifierNames) != len(weightsUp) or len(modifierNames) != len(
                weightsDown
            ):
                raise ValueError(
                    "Provide the same number of modifier names related to the list of modified weights"
                )
        for modifier, weightUp, weightDown in zip(
            modifierNames, weightsUp, weightsDown
        ):
            systName = f"{name}_{modifier}"
            self.__add_variation(systName, weight, weightUp, weightDown, shift)
        self._weightStats[name] = WeightStatistics(
            weight.sum(),
            (weight**2).sum(),
            weight.min(),
            weight.max(),
            weight.size,
        )

    def __add_multivariation_delayed(
        self, name, weight, modifierNames, weightsUp, weightsDown, shift=False
    ):
        """Add a new weight with multiple variations in delayed mode"""
        if isinstance(weight, awkward.types.OptionType):
            # TODO what to do with option-type? is it representative of unknown weight
            # and we default to one or is it an invalid weight and we should never use this
            # event in the first place (0) ?
            weight = dask_awkward.fill_none(weight, 1.0)
        if self._weight is None:
            self._weight = weight
        else:
            self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        # Now loop on the variations
        if len(modifierNames) > 0:
            if len(modifierNames) != len(weightsUp) or len(modifierNames) != len(
                weightsDown
            ):
                raise ValueError(
                    "Provide the same number of modifier names related to the list of modified weights"
                )
        for modifier, weightUp, weightDown in zip(
            modifierNames, weightsUp, weightsDown
        ):
            systName = f"{name}_{modifier}"
            self.__add_variation(systName, weight, weightUp, weightDown, shift)
        self._weightStats[name] = {
            "sumw": dask_awkward.to_dask_array(weight).sum(),
            "sumw2": dask_awkward.to_dask_array(weight**2).sum(),
            "minw": dask_awkward.to_dask_array(weight).min(),
            "maxw": dask_awkward.to_dask_array(weight).max(),
        }

    def add_multivariation(
        self, name, weight, modifierNames, weightsUp, weightsDown, shift=False
    ):
        """Add a new weight with multiple variations

        Each variation of a single weight is given a different modifier name.
        This is particularly useful e.g. for btag SF variations.

        Parameters
        ----------
            name : str
                name of correction
            weight : numpy.ndarray
                the nominal event weight associated with the correction
            modifierNames: list of str
                list of modifiers for each set of weights variation
            weightsUp : list of numpy.ndarray
                weight with correction uncertainty shifted up (if available)
            weightsDown : list of numpy.ndarray
                weight with correction uncertainty shifted down. If ``weightUp`` is supplied, and
                the correction uncertainty is symmetric, this can be set to None to auto-calculate
                the down shift as ``1 / weightUp``.
            shift : bool, optional
                if True, interpret weightUp and weightDown as a relative difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if name.endswith("Up") or name.endswith("Down"):
            raise ValueError(
                "Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call"
            )
        weight = coffea.util._ensure_flat(weight, allow_missing=True)
        if isinstance(weight, numpy.ndarray) and isinstance(
            self._weight, numpy.ndarray
        ):
            self.__add_multivariation_eager(
                name, weight, modifierNames, weightsUp, weightsDown, shift
            )
        elif isinstance(weight, dask_awkward.Array) and isinstance(
            self._weight, (dask_awkward.Array, type(None))
        ):
            self.__add_multivariation_delayed(
                name, weight, modifierNames, weightsUp, weightsDown, shift
            )
        else:
            raise ValueError(
                f"Incompatible weights: self._weight={type(self.weight)}, weight={type(weight)}"
            )

    def __add_variation_eager(self, name, weight, weightUp, weightDown, shift):
        """Helper function to add an eagerly calculated weight variation."""
        if weightUp is not None:
            weightUp = coffea.util._ensure_flat(weightUp, allow_missing=True)
            if isinstance(weightUp, numpy.ma.MaskedArray):
                weightUp = weightUp.filled(1.0)
            if shift:
                weightUp += weight
            weightUp[weight != 0.0] /= weight[weight != 0.0]
            self._modifiers[name + "Up"] = weightUp
        if weightDown is not None:
            weightDown = coffea.util._ensure_flat(weightDown, allow_missing=True)
            if isinstance(weightDown, numpy.ma.MaskedArray):
                weightDown = weightDown.filled(1.0)
            if shift:
                weightDown = weight - weightDown
            weightDown[weight != 0.0] /= weight[weight != 0.0]
            self._modifiers[name + "Down"] = weightDown

    def __add_variation_delayed(self, name, weight, weightUp, weightDown, shift):
        """Helper function to add a delayed-calculation weight variation."""
        if weightUp is not None:
            weightUp = coffea.util._ensure_flat(weightUp, allow_missing=True)
            if isinstance(dask_awkward.type(weightUp), awkward.types.OptionType):
                weightUp = dask_awkward.fill_none(weightUp, 1.0)
            if shift:
                weightUp = weightUp + weight
            weightUp = dask_awkward.where(weight != 0.0, weightUp / weight, weightUp)
            self._modifiers[name + "Up"] = weightUp
        if weightDown is not None:
            weightDown = coffea.util._ensure_flat(weightDown, allow_missing=True)
            if isinstance(dask_awkward.type(weightDown), awkward.types.OptionType):
                weightDown = dask_awkward.fill_none(weightDown, 1.0)
            if shift:
                weightDown = weight - weightDown
            weightDown = dask_awkward.where(
                weight != 0.0, weightDown / weight, weightDown
            )
            self._modifiers[name + "Down"] = weightDown

    def __add_variation(
        self, name, weight, weightUp=None, weightDown=None, shift=False
    ):
        """Helper function to add a weight variation.

        Parameters
        ----------
            name : str
                name of systematic variation (just the name of the weight if only
                one variation is added, or `name_syst` for multiple variations)
            weight : numpy.ndarray
                the nominal event weight associated with the correction
            weightUp : numpy.ndarray, optional
                weight with correction uncertainty shifted up (if available)
            weightDown : numpy.ndarray, optional
                weight with correction uncertainty shifted down. If ``weightUp`` is supplied, and
                the correction uncertainty is symmetric, this can be set to None to auto-calculate
                the down shift as ``1 / weightUp``.
            shift : bool, optional
                if True, interpret weightUp and weightDown as a relative difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if isinstance(weight, numpy.ndarray):
            self.__add_variation_eager(name, weight, weightUp, weightDown, shift)
        elif isinstance(weight, dask_awkward.Array):
            self.__add_variation_delayed(name, weight, weightUp, weightDown, shift)

    def weight(self, modifier=None):
        """Current event weight vector

        Parameters
        ----------
            modifier : str, optional
                if supplied, provide event weight corresponding to a particular
                systematic uncertainty shift, of form ``str(name + 'Up')`` or (Down)

        Returns
        -------
            weight : numpy.ndarray
                The weight vector, possibly modified by the effect of a given systematic variation.
        """
        if modifier is None:
            return self._weight
        elif "Down" in modifier and modifier not in self._modifiers:
            return self._weight / self._modifiers[modifier.replace("Down", "Up")]
        return self._weight * self._modifiers[modifier]

    def partial_weight(self, include=[], exclude=[]):
        """Partial event weight vector

        Return a partial weight by multiplying a subset of all weights.
        Can be operated either by specifying weights to include or
        weights to exclude, but not both at the same time. The method
        can only be used if the individual weights are stored via the
        ``storeIndividual`` argument in the `Weights` initializer.

        Parameters
        ----------
            include : list
                Weight names to include, defaults to []
            exclude : list
                Weight names to exclude, defaults to []
        Returns
        -------
            weight : numpy.ndarray
                The weight vector, corresponding to only the effect of the
                corrections specified.
        """
        if not self._storeIndividual:
            raise ValueError(
                "To be able to request weight exclusion, use storeIndividual=True when creating Weights object."
            )
        if (include and exclude) or not (include or exclude):
            raise ValueError(
                "Need to specify exactly one of the 'exclude' or 'include' arguments."
            )

        names = set(self._weights.keys())
        if include:
            names = names & set(include)
        if exclude:
            names = names - set(exclude)

        w = None
        if isinstance(self._weight, numpy.ndarray):
            w = numpy.ones(self._weight.size)
        elif isinstance(self._weight, dask_awkward.Array):
            w = dask_awkward.ones_like(self._weight)

        for name in names:
            w = w * self._weights[name]

        return w

    @property
    def variations(self):
        """List of available modifiers"""
        keys = set(self._modifiers.keys())
        # add any missing 'Down' variation
        for k in self._modifiers.keys():
            keys.add(k.replace("Up", "Down"))
        return keys


class NminusOne:
    """Object to be returned by PackedSelection.nminusone()"""

    def __init__(self, names, nev, masks, delayed_mode):
        self._names = names
        self._nev = nev
        self._masks = masks
        self._delayed_mode = delayed_mode

    def __repr__(self):
        return f"NminusOne(selections={self._names})"

    def result(self):
        """Returns the results of the N-1 selection as a namedtuple

        Returns
        -------
            result : NminusOneResult
                A namedtuple with the following attributes:

                nev : list of integers or dask_awkward.lib.core.Scalar objects
                    The number of events in each step of the N-1 selection as a list of integers or delayed integers
                masks : list of boolean numpy.ndarray or dask_awkward.lib.core.Array objects
                    The boolean mask vectors of which events pass the N-1 selection each time as a list of materialized or delayed boolean arrays

        """
        NminusOneResult = namedtuple("NminusOneResult", ["labels", "nev", "masks"])
        labels = ["initial"] + [f"N - {i}" for i in self._names] + ["N"]
        return NminusOneResult(labels, self._nev, self._masks)

    def print(self):
        """Prints the statistics of the N-1 selection"""

        if self._delayed_mode:
            self._nev = list(dask.compute(*self._nev))
        nev = self._nev
        print("N-1 selection stats:")
        for i, name in enumerate(self._names):
            print(
                f"Ignoring {name:<20}: pass = {nev[i+1]:<20}\
                all = {nev[0]:<20}\
                -- eff = {nev[i+1]*100/nev[0]:.1f} %"
            )

        if True:
            print(
                f"All cuts {'':<20}: pass = {nev[-1]:<20}\
                all = {nev[0]:<20}\
                -- eff = {nev[-1]*100/nev[0]:.1f} %"
            )

    def yieldhist(self):
        """Returns the N-1 selection yields as a ``hist.Hist`` object

        Returns
        -------
            h : hist.Hist or hist.dask.Hist
                Histogram of the number of events surviving the N-1 selection
            labels : list of strings
                The bin labels of the histogram
        """
        labels = ["initial"] + [f"N - {i}" for i in self._names] + ["N"]
        if not self._delayed_mode:
            h = hist.Hist(
                hist.axis.IntCategory(
                    numpy.arange(len(labels)), growth=True, name="N-1"
                )
            )
            h.fill(numpy.arange(len(labels)), weight=self._nev)

        elif self._delayed_mode:
            h = hist.dask.Hist(
                hist.axis.IntCategory(
                    numpy.arange(len(labels)), growth=True, name="N-1"
                )
            )
            for i, weight in enumerate(self._masks, 1):
                h.fill(dask_awkward.full_like(weight, i, dtype=int), weight=weight)
            h.fill(
                dask_awkward.from_awkward(awkward.Array([0]), 1),
                weight=dask_awkward.from_awkward(awkward.Array([len(weight)]), 1),
            )
        return h, labels

    def plot_vars(self, vars):
        """Plot the histograms of variables for each step of the N-1 selection

        Parameters
        ----------
            vars : dict
                A dictionary in the form ``{name: array}`` where ``name`` is the name of the variable,
                and ``array`` is the corresponding array of values

        Returns
        -------
            hists : list of hist.Hist or hist.dask.Hist objects
                A list of 2D histograms of the variables for each step of the N-1 selection.
                The first axis is the variable, the second axis is the N-1 selection step.
            labels : list of strings
                The bin labels of y axis of the histogram
        """

        hists = []
        labels = ["initial"] + [f"N - {i}" for i in self._names] + ["N"]

        if not self._delayed_mode:
            for name, var in vars.items():
                h = hist.Hist(
                    hist.axis.Regular(
                        20,
                        awkward.min(var) - 1e-5,
                        awkward.max(var) + 1e-5,
                        name=name,
                    ),
                    hist.axis.IntCategory(numpy.arange(len(labels)), name="N-1"),
                )
                arr = awkward.flatten(var)
                h.fill(arr, awkward.zeros_like(arr))
                for i, mask in enumerate(self.result().masks, 1):
                    arr = awkward.flatten(var[mask])
                    h.fill(arr, awkward.full_like(arr, i, dtype=int))
                hists.append(h)

        elif self._delayed_mode:
            for name, var in vars.items():
                h = hist.dask.Hist(
                    hist.axis.Regular(
                        20,
                        dask_awkward.min(var).compute() - 1e-5,
                        dask_awkward.max(var).compute() + 1e-5,
                        name=name,
                    ),
                    hist.axis.IntCategory(numpy.arange(len(labels)), name="N-1"),
                )
                arr = dask_awkward.flatten(var)
                h.fill(arr, dask_awkward.zeros_like(arr))
                for i, mask in enumerate(self.result().masks, 1):
                    arr = dask_awkward.flatten(var[mask])
                    h.fill(arr, dask_awkward.full_like(arr, i, dtype=int))
                hists.append(h)

        return hists, labels


class Cutflow:
    """Object to be returned by PackedSelection.cutflow()"""

    def __init__(
        self, names, nevonecut, nevcutflow, masksonecut, maskscutflow, delayed_mode
    ):
        self._names = names
        self._nevonecut = nevonecut
        self._nevcutflow = nevcutflow
        self._masksonecut = masksonecut
        self._maskscutflow = maskscutflow
        self._delayed_mode = delayed_mode

    def __repr__(self):
        return f"Cutflow(selections={self._names})"

    def result(self):
        """Returns the results of the cutflow as a namedtuple

        Returns
        -------
            result : CutflowResult
                A namedtuple with the following attributes:

                nevonecut : list of integers or dask_awkward.lib.core.Scalar objects
                    The number of events that survive each cut alone as a list of integers or delayed integers
                nevcutflow : list of integers or dask_awkward.lib.core.Scalar objects
                    The number of events that survive the cumulative cutflow as a list of integers or delayed integers
                masksonecut : list of boolean numpy.ndarray or dask_awkward.lib.core.Array objects
                    The boolean mask vectors of which events pass each cut alone as a list of materialized or delayed boolean arrays
                maskscutflow : list of boolean numpy.ndarray or dask_awkward.lib.core.Array objects
                    The boolean mask vectors of which events pass the cumulative cutflow a list of materialized or delayed boolean arrays
        """
        CutflowResult = namedtuple(
            "CutflowResult",
            ["labels", "nevonecut", "nevcutflow", "masksonecut", "maskscutflow"],
        )
        labels = ["initial"] + list(self._names)
        return CutflowResult(
            labels,
            self._nevonecut,
            self._nevcutflow,
            self._masksonecut,
            self._maskscutflow,
        )

    def print(self):
        """Prints the statistics of the Cutflow"""

        if self._delayed_mode:
            self._nevonecut = list(dask.compute(*self._nevonecut))
            self._nevcutflow = list(dask.compute(*self._nevcutflow))
        nevonecut = self._nevonecut
        nevcutflow = self._nevcutflow
        print("Cutflow stats:")
        for i, name in enumerate(self._names):
            print(
                f"Cut {name:<20}: pass = {nevonecut[i+1]:<20}\
                cumulative pass = {nevcutflow[i+1]:<20}\
                all = {nevonecut[0]:<20}\
                --  eff = {nevonecut[i+1]*100/nevonecut[0]:.1f} %\
                -- cumulative eff = {nevcutflow[i+1]*100/nevcutflow[0]:.1f} %"
            )

    def yieldhist(self):
        """Returns the cutflow yields as ``hist.Hist`` objects

        Returns
        -------
            honecut : hist.Hist or hist.dask.Hist
                Histogram of the number of events surviving each cut alone
            hcutflow : hist.Hist or hist.dask.Hist
                Histogram of the number of events surviving the cumulative cutflow
            labels : list of strings
                The bin labels of the histograms
        """
        labels = ["initial"] + list(self._names)

        if not self._delayed_mode:
            honecut = hist.Hist(
                hist.axis.IntCategory(
                    numpy.arange(len(labels)), growth=True, name="onecut"
                )
            )
            hcutflow = honecut.copy()
            hcutflow.axes.name = ("cutflow",)
            honecut.fill(numpy.arange(len(labels)), weight=self._nevonecut)
            hcutflow.fill(numpy.arange(len(labels)), weight=self._nevcutflow)

        elif self._delayed_mode:
            honecut = hist.dask.Hist(
                hist.axis.IntCategory(
                    numpy.arange(len(labels)), growth=True, name="onecut"
                )
            )
            hcutflow = honecut.copy()
            hcutflow.axes.name = ("cutflow",)

            for i, weight in enumerate(self._masksonecut, 1):
                honecut.fill(
                    dask_awkward.full_like(weight, i, dtype=int), weight=weight
                )
            honecut.fill(
                dask_awkward.from_awkward(awkward.Array([0]), 1),
                weight=dask_awkward.from_awkward(awkward.Array([len(weight)]), 1),
            )
            for i, weight in enumerate(self._maskscutflow, 1):
                hcutflow.fill(
                    dask_awkward.full_like(weight, i, dtype=int), weight=weight
                )
            hcutflow.fill(
                dask_awkward.from_awkward(awkward.Array([0]), 1),
                weight=dask_awkward.from_awkward(awkward.Array([len(weight)]), 1),
            )

        return honecut, hcutflow, labels

    def plot_vars(self, vars):
        """Plot the histograms of variables for each step of the N-1 selection

        Parameters
        ----------
            vars : dict
                A dictionary in the form ``{name: array}`` where ``name`` is the name of the variable,
                and ``array`` is the corresponding array of values

        Returns
        -------
            histsonecut : list of hist.Hist or hist.dask.Hist objects
                A list of 1D histograms of the variables of events surviving each cut alone.
                The first axis is the variable, the second axis is the cuts.
            histscutflow : list of hist.Hist or hist.dask.Hist objects
                A list of 1D histograms of the variables of events surviving the cumulative cutflow.
                The first axis is the variable, the second axis is the cuts.
            labels : list of strings
                The bin labels of the y axis of the histograms
        """

        histsonecut, histscutflow = [], []
        labels = ["initial"] + list(self._names)

        if not self._delayed_mode:
            for name, var in vars.items():
                honecut = hist.Hist(
                    hist.axis.Regular(
                        20,
                        awkward.min(var) - 1e-5,
                        awkward.max(var) + 1e-5,
                        name=name,
                    ),
                    hist.axis.IntCategory(numpy.arange(len(labels)), name="onecut"),
                )
                hcutflow = honecut.copy()
                hcutflow.axes.name = name, "cutflow"

                arr = awkward.flatten(var)
                honecut.fill(arr, awkward.zeros_like(arr))
                hcutflow.fill(arr, awkward.zeros_like(arr))

                for i, mask in enumerate(self.result().masksonecut, 1):
                    arr = awkward.flatten(var[mask])
                    honecut.fill(arr, awkward.full_like(arr, i, dtype=int))
                histsonecut.append(honecut)

                for i, mask in enumerate(self.result().maskscutflow, 1):
                    arr = awkward.flatten(var[mask])
                    hcutflow.fill(arr, awkward.full_like(arr, i, dtype=int))
                histscutflow.append(hcutflow)

        elif self._delayed_mode:
            for name, var in vars.items():
                honecut = hist.dask.Hist(
                    hist.axis.Regular(
                        20,
                        dask_awkward.min(var).compute() - 1e-5,
                        dask_awkward.max(var).compute() + 1e-5,
                        name=name,
                    ),
                    hist.axis.IntCategory(numpy.arange(len(labels)), name="onecut"),
                )
                hcutflow = honecut.copy()
                hcutflow.axes.name = name, "cutflow"

                arr = dask_awkward.flatten(var)
                honecut.fill(arr, dask_awkward.zeros_like(arr))
                hcutflow.fill(arr, dask_awkward.zeros_like(arr))

                for i, mask in enumerate(self.result().masksonecut, 1):
                    arr = dask_awkward.flatten(var[mask])
                    honecut.fill(arr, dask_awkward.full_like(arr, i, dtype=int))
                histsonecut.append(honecut)

                for i, mask in enumerate(self.result().maskscutflow, 1):
                    arr = dask_awkward.flatten(var[mask])
                    hcutflow.fill(arr, dask_awkward.full_like(arr, i, dtype=int))
                histscutflow.append(hcutflow)

        return histsonecut, histscutflow, labels


class PackedSelection:
    """Store several boolean arrays in a compact manner

    This class can store several boolean arrays in a memory-efficient mannner
    and evaluate arbitrary combinations of boolean requirements in an CPU-efficient way.
    Supported inputs are 1D numpy or awkward arrays.

    Parameters
    ----------
        dtype : numpy.dtype or str
            internal bitwidth of the packed array, which governs the maximum
            number of selections storable in this object. The default value
            is ``uint32``, which allows up to 32 booleans to be stored, but
            if a smaller or larger number of selections needs to be stored,
            one can choose ``uint16`` or ``uint64`` instead.
    """

    _supported_types = {
        numpy.dtype("uint16"): 16,
        numpy.dtype("uint32"): 32,
        numpy.dtype("uint64"): 64,
    }

    def __init__(self, dtype="uint32"):
        self._dtype = numpy.dtype(dtype)
        if self._dtype not in PackedSelection._supported_types:
            raise ValueError(f"dtype {dtype} is not supported")
        self._names = []
        self._data = None

    def __repr__(self):
        delayed_mode = None if self._data is None else self.delayed_mode
        return f"PackedSelection(selections={tuple(self._names)}, delayed_mode={delayed_mode}, items={len(self._names)}, maxitems={self.maxitems})"

    @property
    def names(self):
        """Current list of mask names available"""
        return self._names

    @property
    def delayed_mode(self):
        if isinstance(self._data, dask_awkward.Array):
            return True
        elif isinstance(self._data, numpy.ndarray):
            return False
        else:
            return "PackedSelection hasn't been initialized with a boolean array yet!"

    @property
    def maxitems(self):
        return PackedSelection._supported_types[self._dtype]

    def __add_delayed(self, name, selection, fill_value):
        """Add a new delayed boolean array"""
        selection = coffea.util._ensure_flat(selection, allow_missing=True)
        sel_type = dask_awkward.type(selection)
        if isinstance(sel_type, awkward.types.OptionType):
            selection = dask_awkward.fill_none(selection, fill_value)
            sel_type = dask_awkward.type(selection)
        if sel_type.primitive != "bool":
            raise ValueError(f"Expected a boolean array, received {sel_type.primitive}")
        if len(self._names) == 0:
            self._data = dask_awkward.zeros_like(selection, dtype=self._dtype)
        if isinstance(selection, dask_awkward.Array) and not self.delayed_mode:
            raise ValueError(
                f"New selection '{name}' is not eager while PackedSelection is!"
            )
        elif len(self._names) == self.maxitems:
            raise RuntimeError(
                f"Exhausted all slots in this PackedSelection, consider a larger dtype or fewer selections"
            )
        elif not dask_awkward.lib.core.compatible_partitions(self._data, selection):
            raise ValueError(
                f"New selection '{name}' has a different partition structure than existing selections"
            )
        self._data = numpy.bitwise_or(
            self._data,
            selection * self._dtype.type(1 << len(self._names)),
        )
        self._names.append(name)

    def __add_eager(self, name, selection, fill_value):
        """Add a new eager boolean array"""
        selection = coffea.util._ensure_flat(selection, allow_missing=True)
        if isinstance(selection, numpy.ma.MaskedArray):
            selection = selection.filled(fill_value)
        if selection.dtype != bool:
            raise ValueError(f"Expected a boolean array, received {selection.dtype}")
        if len(self._names) == 0:
            self._data = numpy.zeros(len(selection), dtype=self._dtype)
        if isinstance(selection, numpy.ndarray) and self.delayed_mode:
            raise ValueError(
                f"New selection '{name}' is not delayed while PackedSelection is!"
            )
        elif len(self._names) == self.maxitems:
            raise RuntimeError(
                "Exhausted all slots in this PackedSelection, consider a larger dtype or fewer selections"
            )
        elif self._data.shape != selection.shape:
            raise ValueError(
                f"New selection '{name}' has a different shape than existing selections ({selection.shape} vs. {self._data.shape})"
            )
        numpy.bitwise_or(
            self._data,
            self._dtype.type(1 << len(self._names)),
            where=selection,
            out=self._data,
        )
        self._names.append(name)

    def add(self, name, selection, fill_value=False):
        """Add a new boolean array

        Parameters
        ----------
            name : str
                name of the selection
            selection : numpy.ndarray or awkward.Array
                a flat array of type ``bool`` or ``?bool``.
                If this is not the first selection added, it must also have
                the same shape as previously added selections. If the array
                is option-type, null entries will be filled with ``fill_value``.
            fill_value : bool, optional
                All masked entries will be filled as specified (default: ``False``)
        """
        if isinstance(selection, dask.array.Array):
            raise ValueError(
                "Dask arrays are not supported, please convert them to dask_awkward.Array by using dask_awkward.from_dask_array()"
            )
        selection = coffea.util._ensure_flat(selection, allow_missing=True)
        if isinstance(selection, numpy.ndarray):
            self.__add_eager(name, selection, fill_value)
        elif isinstance(selection, dask_awkward.Array):
            self.__add_delayed(name, selection, fill_value)

    def add_multiple(self, selections, fill_value=False):
        """Add multiple boolean arrays at once, see ``add`` for details

        Parameters
        ----------
            selections : dict
                a dictionary of selections, in the form ``{name: selection}``
            fill_value : bool, optional
                All masked entries will be filled as specified (default: ``False``)
        """
        for name, selection in selections.items():
            self.add(name, selection, fill_value)

    def require(self, **names):
        """Return a mask vector corresponding to specific requirements

        Specify an exact requirement on an arbitrary subset of the masks

        Parameters
        ----------
            ``**names`` : kwargs
                Each argument to require specific value for, in form ``arg=True``
                or ``arg=False``.

        Examples
        --------
        If

        >>> selection.names
        ['cut1', 'cut2', 'cut3']

        then

        >>> selection.require(cut1=True, cut2=False)
        array([True, False, True, ...])

        returns a boolean array where an entry is True if the corresponding entries
        ``cut1 == True``, ``cut2 == False``, and ``cut3`` arbitrary.
        """
        consider = 0
        require = 0
        for name, val in names.items():
            val = bool(val)
            idx = self._names.index(name)
            consider |= 1 << idx
            require |= int(val) << idx
        return (self._data & consider) == require

    def all(self, *names):
        """Shorthand for `require`, where all the values are True"""
        return self.require(**{name: True for name in names})

    def any(self, *names):
        """Return a mask vector corresponding to an inclusive OR of requirements

        Parameters
        ----------
            ``*names`` : args
                The named selections to allow

        Examples
        --------
        If

        >>> selection.names
        ['cut1', 'cut2', 'cut3']

        then

        >>> selection.any("cut1", "cut2")
        array([True, False, True, ...])

        returns a boolean array where an entry is True if the corresponding entries
        ``cut1 == True`` or ``cut2 == False``, and ``cut3`` arbitrary.
        """
        consider = 0
        for name in names:
            idx = self._names.index(name)
            consider |= 1 << idx
        return (self._data & consider) != 0

    def nminusone(self, *names):
        """Compute the "N-1" style selection for a set of selections

        The N-1 style selection for a set of selections, returns an object which can return a list of the number of events
        that pass all the other selections ignoring one at a time. The first element of the returned list
        is the total number of events before any selections are applied.
        The last element is the final number of events that pass if all selections are applied.
        It also returns a list of boolean mask vectors of which events pass the N-1 selection each time.
        Can also return a histogram as a ``hist.Hist`` object where the bin heights are the number of events of the N-1 selection list.
        If the PackedSelection is in delayed mode, the elements of those lists will be dask_awkward Arrays that can be computed whenever the user wants.
        If the histogram is requested, the delayed arrays of the number of events list will be computed in the process in order to set the bin heights.

        Parameters
        ----------
            ``*names`` : args
                The named selections to use, need to be a subset of the selections already added

        Returns
        -------
            res: coffea.analysis_tools.NminusOne
                A wrapper class for the results, see the documentation for that class for more details
        """
        for cut in names:
            if not isinstance(cut, str) or cut not in self.names:
                raise ValueError(
                    "All arguments must be strings that refer to the names of existing selections"
                )

        masks = []
        for i, cut in enumerate(names):
            mask = self.all(*(names[:i] + names[i + 1 :]))
            masks.append(mask)
        mask = self.all(*names)
        masks.append(mask)

        if not self.delayed_mode:
            nev = [len(self._data)]
            nev.extend(numpy.sum(masks, axis=1))

        elif self.delayed_mode:
            nev = [
                dask_awkward.sum(
                    dask_awkward.from_awkward(awkward.Array([len(self._data)]), 1)
                )
            ]
            nev.extend([dask_awkward.sum(mask) for mask in masks])

        return NminusOne(names, nev, masks, self.delayed_mode)

    def cutflow(self, *names):
        """Compute the cutflow for a set of selections

        Returns an object which can return a list of the number of events that pass all the previous selections including the current one
        after each named selection is applied consecutively. The first element
        of the returned list is the total number of events before any selections are applied.
        The last element is the final number of events that pass after all the selections are applied.
        Can also return a cutflow histogram as a ``hist.Hist`` object where the bin heights are the number of events of the cutflow list.
        If the PackedSelection is in delayed mode, the elements of the list will be dask_awkward Arrays that can be computed whenever the user wants.
        If the histogram is requested, those delayed arrays will be computed in the process in order to set the bin heights.

        Parameters
        ----------
            ``*names`` : args
                The named selections to use, need to be a subset of the selections already added

        Returns
        -------
            res: coffea.analysis_tools.Cutflow
                A wrapper class for the results, see the documentation for that class for more details
        """
        for cut in names:
            if not isinstance(cut, str) or cut not in self.names:
                raise ValueError(
                    "All arguments must be strings that refer to the names of existing selections"
                )

        masksonecut, maskscutflow = [], []
        for i, cut in enumerate(names):
            mask1 = self.any(cut)
            mask2 = self.all(*(names[: i + 1]))
            masksonecut.append(mask1)
            maskscutflow.append(mask2)

        if not self.delayed_mode:
            nevonecut = [len(self._data)]
            nevcutflow = [len(self._data)]
            nevonecut.extend(numpy.sum(masksonecut, axis=1))
            nevcutflow.extend(numpy.sum(maskscutflow, axis=1))

        elif self.delayed_mode:
            nevonecut = [
                dask_awkward.sum(
                    dask_awkward.from_awkward(awkward.Array([len(self._data)]), 1)
                )
            ]
            nevcutflow = [
                dask_awkward.sum(
                    dask_awkward.from_awkward(awkward.Array([len(self._data)]), 1)
                )
            ]
            nevonecut.extend([dask_awkward.sum(mask1) for mask1 in masksonecut])
            nevcutflow.extend([dask_awkward.sum(mask2) for mask2 in maskscutflow])

        return Cutflow(
            names, nevonecut, nevcutflow, masksonecut, maskscutflow, self.delayed_mode
        )
