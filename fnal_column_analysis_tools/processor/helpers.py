from ..util import numpy as np


class Weights(object):
    """
    Keep track of event weight corrections, as well as any accompanying systematic
    shifts via multiplicative modifiers.
    """
    def __init__(self, size):
        self._weight = np.ones(size)
        self._modifiers = {}
        self._weightStats = {}

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
    """
    def __init__(self, dtype='uint64'):
        """
            dtype: internal bitwidth, numpy supports up to uint64
                    smaller values may be more efficient
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
