from ..util import numpy as np


class Weights(object):
    """
    Keep track of event weight corrections, as well as any accompanying systematic
    shifts via multiplicative modifiers.
    """
    def __init__(self, size, storeIndividual=False):
        """
        Initialization.

        Parameters
        ----------
            size:
                Size of the weight arrays to be handled (i.e. the number of events / instances).
            storeIndividual:
                Store not only the total weight + variations, but also each individual weight.
        """
        self._weight = np.ones(size)
        self._weights = {}
        self._modifiers = {}
        self._weightStats = {}
        self._storeIndividual = storeIndividual

    def add(self, name, weight, weightUp=None, weightDown=None, shift=False):
        """
        Add a correction to the overall event weight, and keep track of systematic uncertainties
        if they exist.
            name: name of correction weight
            weight: nominal weight
            weightUp: weight with correction uncertainty shifted up (if available)
            weightDown: weight with correction uncertainty shifted down (leave None if symmetric)
            shift: if True, interpret weightUp and weightDown as a difference relative to the nominal value
        """
        if 'Up' in name or 'Down' in name:
            raise ValueError("Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call")
        self._weight *= weight
        if self._storeIndividual:
            self._weights[name] = weight
        if weightUp is not None:
            if shift:
                weightUp += weight
            weightUp[weight != 0.] /= weight[weight != 0.]
            self._modifiers[name + 'Up'] = weightUp
        if weightDown is not None:
            if shift:
                weightDown = weight - weightDown
            weightDown[weight != 0.] /= weight[weight != 0.]
            self._modifiers[name + 'Down'] = weightDown
        self._weightStats[name] = {
            'sumw': weight.sum(),
            'sumw2': (weight**2).sum(),
            'min': weight.min(),
            'max': weight.max(),
            'n': weight.size,
        }

    def weight(self, modifier=None):
        """
        Return a weight, possibly modified by a systematic variation
        """
        if modifier is None:
            return self._weight
        elif 'Down' in modifier and modifier not in self._modifiers:
            return self._weight / self._modifiers[modifier.replace('Down', 'Up')]
        return self._weight * self._modifiers[modifier]

    def partial_weight(self, include=[], exclude=[]):
        """
        Return a partial weight by multiplying a subset of all weights.

        Can be operated either by specifying weights to include or
        weights to exclude, but not both at the same time. The method
        can only be used if the individual weights are stored (see
        storeIndividual argument in the initializer).

        Parameters
        ----------
            include:
                Weight names to include, defaults to []
            exclude:
                Weight names to exclude, defaults to []
        """
        if not self._storeIndividual:
            raise ValueError("To be able to request weight exclusion, use storeIndividual=True when creating Weights object.")
        if (include and exclude) or not (include or exclude):
            raise ValueError("Need to specify exactly one of the 'exclude' or 'include' arguments.")

        names = set(self._weights.keys())
        if include:
            names = names & set(include)
        if exclude:
            names = names - set(exclude)

        w = np.ones(self._weight.size)
        for name in names:
            w = w * self._weights[name]

        return w

    @property
    def variations(self):
        """
        List of modifiers (systematic variations) available
        """
        keys = set(self._modifiers.keys())
        # add any missing 'Down' variation
        for k in self._modifiers.keys():
            keys.add(k.replace('Up', 'Down'))
        return keys


class PackedSelection(object):
    """
    Stores several boolean masks (cuts, selections) in a compact manner
    and evaluates arbitrary requirements on each in an CPU-efficient way

    Parameters
    ----------
        dtype:
            internal bitwidth, numpy supports up to uint64
            smaller values may be more efficient
    """
    def __init__(self, dtype='uint64'):
        """
        TODO: extend to multi-column for arbitrary bit depth
        """
        self._dtype = np.dtype(dtype)
        self._names = []
        self._mask = None

    @property
    def names(self):
        return self._names

    def add(self, name, selection):
        """
        Add a selection to the set of masks
            name: obvious
            selection: must be a numpy flat array of dtype bool
                        if not the first selection added, it must also have
                        the same shape as previous inputs
        """
        if isinstance(selection, np.ndarray) and selection.dtype == np.dtype('bool'):
            if len(self._names) == 0:
                self._mask = np.zeros(shape=selection.shape, dtype=self._dtype)
            elif len(self._names) == 64:
                raise RuntimeError("Exhausted all slots for %r, consider a larger dtype or fewer selections" % self._dtype)
            elif self._mask.shape != selection.shape:
                raise ValueError("New selection '%s' has different shape than existing ones (%r vs. %r)" % (name, selection.shape, self._mask.shape))
            self._mask |= selection.astype(self._dtype) << len(self._names)
            self._names.append(name)
        else:
            raise ValueError("PackedSelection only understands numpy boolean arrays, got %r" % selection)

    def require(self, **names):
        """
        Specify an exact requirement on an arbitrary subset of the masks
        e.g. if selection.names = ['cut1', 'cut2', 'cut3'], then:
        selection.require(cut1=True, cut2=False) returns a boolean mask where
        an event passes if cut1 is True, cut2 is False, and cut3 is arbitrary
        """
        mask = 0
        require = 0
        for name, val in names.items():
            if not isinstance(val, bool):
                raise ValueError("Please use only booleans in PackedSelection.require(), received %r for %s" % (val, name))
            idx = self._names.index(name)
            mask |= 1 << idx
            require |= int(val) << idx
        return (self._mask & mask) == require

    def all(self, *names):
        """
        Shorthand for require, where all the values must be True
        """
        return self.require(**{name: True for name in names})
