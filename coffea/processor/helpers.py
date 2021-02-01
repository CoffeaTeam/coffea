from coffea.util import deprecate
import numpy


class Weights(object):
    """Container for event weights and associated systematic shifts

    This container keeps track of correction factors and systematic
    effects that can be encoded as multiplicative modifiers to the event weight.
    All weights are stored in vector form.

    Parameters
    ----------
        size : int
            size of the weight arrays to be handled (i.e. the number of events / instances).
        storeIndividual : bool, optional
            store not only the total weight + variations, but also each individual weight.
            Default is false.
    """
    def __init__(self, size, storeIndividual=False):
        deprecate("This utility has moved to the `coffea.analysis_tools` subpackage and has new features, check it out!", 0.8)
        self._weight = numpy.ones(size)
        self._weights = {}
        self._modifiers = {}
        self._weightStats = {}
        self._storeIndividual = storeIndividual

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
                if True, interpret weightUp and weightDown as a realtive difference (additive) to the
                nominal value

        .. note:: ``weightUp`` and ``weightDown`` are assumed to be rvalue-like and may be modified in-place by this function
        """
        if name.endswith('Up') or name.endswith('Down'):
            raise ValueError("Avoid using 'Up' and 'Down' in weight names, instead pass appropriate shifts to add() call")
        weight = numpy.array(weight)
        self._weight = self._weight * weight
        if self._storeIndividual:
            self._weights[name] = weight
        if weightUp is not None:
            weightUp = numpy.array(weightUp)
            if shift:
                weightUp += weight
            weightUp[weight != 0.] /= weight[weight != 0.]
            self._modifiers[name + 'Up'] = weightUp
        if weightDown is not None:
            weightDown = numpy.array(weightDown)
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
        elif 'Down' in modifier and modifier not in self._modifiers:
            return self._weight / self._modifiers[modifier.replace('Down', 'Up')]
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
            raise ValueError("To be able to request weight exclusion, use storeIndividual=True when creating Weights object.")
        if (include and exclude) or not (include or exclude):
            raise ValueError("Need to specify exactly one of the 'exclude' or 'include' arguments.")

        names = set(self._weights.keys())
        if include:
            names = names & set(include)
        if exclude:
            names = names - set(exclude)

        w = numpy.ones(self._weight.size)
        for name in names:
            w = w * self._weights[name]

        return w

    @property
    def variations(self):
        """List of available modifiers"""
        keys = set(self._modifiers.keys())
        # add any missing 'Down' variation
        for k in self._modifiers.keys():
            keys.add(k.replace('Up', 'Down'))
        return keys


class PackedSelection(object):
    """Store boolean mask vectors in a compact manner

    This class can store several boolean masks (cuts, selections) and
    evaluate arbitrary combinations of the requirements in an CPU-efficient way

    Parameters
    ----------
        dtype : str
            internal bitwidth of mask vector, which governs the maximum
            number of boolean masks storable in this object.
            By default, up to 64 masks can be stored, but smaller values
            for the `numpy.dtype` may be more efficient.
    """
    def __init__(self, dtype='uint64'):
        """
        TODO: extend to multi-column for arbitrary bit depth
        """
        deprecate("This utility has moved to the `coffea.analysis_tools` subpackage and has new features, check it out!", 0.8)
        self._dtype = numpy.dtype(dtype)
        self._names = []
        self._mask = None

    @property
    def names(self):
        """Current list of mask names available"""
        return self._names

    def add(self, name, selection):
        """Add a named mask

        Parameters
        ----------
            name : str
                name of the mask
            selection : numpy.ndarray
                a flat array of dtype bool.
                If not the first mask added, it must also have
                the same shape as previously added masks.
        """
        if isinstance(selection, numpy.ndarray) and selection.dtype == numpy.dtype('bool'):
            if len(self._names) == 0:
                self._mask = numpy.zeros(shape=selection.shape, dtype=self._dtype)
            elif len(self._names) == 64:
                raise RuntimeError("Exhausted all slots for %r, consider a larger dtype or fewer selections" % self._dtype)
            elif self._mask.shape != selection.shape:
                raise ValueError("New selection '%s' has different shape than existing ones (%r vs. %r)" % (name, selection.shape, self._mask.shape))
            self._mask |= selection.astype(self._dtype) << len(self._names)
            self._names.append(name)
        else:
            raise ValueError("PackedSelection only understands numpy boolean arrays, got %r" % selection)

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

        returns a boolean array where each entry passes if the corresponding entry has
        ``cut1 == True``, ``cut2 == False``, and ``cut3`` arbitrary.
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
        """Shorthand for `require`, where all the values are True
        """
        return self.require(**{name: True for name in names})
