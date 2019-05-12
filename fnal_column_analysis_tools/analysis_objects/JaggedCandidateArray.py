import uproot_methods

import math
from fnal_column_analysis_tools.util import awkward
from fnal_column_analysis_tools.util import numpy as np

JaggedTLorentzVectorArray = awkward.Methods.mixin(uproot_methods.classes.TLorentzVector.ArrayMethods, awkward.JaggedArray)


# functions to quickly cash useful quantities
def _fast_pt(p4):
    """ quick pt calculation for caching """
    return np.hypot(p4.x, p4.y)


def _fast_eta(p4):
    """ quick eta calculation for caching """
    px = p4.x
    py = p4.y
    pz = p4.z
    pT = np.sqrt(px * px + py * py)
    return np.arcsinh(pz / pT)


def _fast_phi(p4):
    """ quick phi calculation for caching """
    return np.arctan2(p4.y, p4.x)


def _fast_mass(p4):
    """ quick mass calculation for caching """
    px = p4.x
    py = p4.y
    pz = p4.z
    en = p4.t
    p3mag2 = (px * px + py * py + pz * pz)
    return np.sqrt(np.abs(en * en - p3mag2))


def _default_match(combs, deltaRCut=10000, deltaPtCut=10000):
    """ default matching function for match(), match in deltaR / deltaPt """
    passPtCut = ((np.abs(combs.i0.pt - combs.i1.pt) / combs.i0.pt) < deltaPtCut)
    mask = (combs.i0.delta_r(combs.i1) < deltaRCut) & passPtCut
    return mask.any()


def _default_argmatch(combs, deltaRCut=10000, deltaPtCut=10000):
    """ default matching function for argmatch(), match in deltaR / deltaPt """
    deltaPts = (np.abs(combs.i0.pt - combs.i1.pt) / combs.i0.pt)
    deltaRs = combs.i0.delta_r(combs.i1)
    indexOfMin = deltaRs.argmin()
    indexOfMinOutShape = indexOfMin.flatten(axis=1)
    passesCut = (deltaRs[indexOfMin] < deltaRCut) & (deltaPts[indexOfMin] < deltaPtCut)
    passesCutOutShape = passesCut.flatten(axis=1)
    flatPass = passesCutOutShape.flatten()
    flatIdxMin = indexOfMinOutShape.flatten()
    flatIdxMin[~flatPass] = -1
    return awkward.JaggedArray.fromoffsets(passesCutOutShape.offsets, flatIdxMin)


def _default_fastmatch(first, second, deltaRCut=10000):
    drCut2 = deltaRCut**2
    args = first.eta._argcross(second.eta)
    argsnested = awkward.JaggedArray.fromcounts(first.eta.counts,
                                                awkward.JaggedArray.fromcounts(first.eta._broadcast(second.eta.counts).flatten(),
                                                                               args._content))
    eta0s = first.eta.content[argsnested.content.content.i0]
    eta1s = second.eta.content[argsnested.content.content.i1]
    phi0s = first.phi.content[argsnested.content.content.i0]
    phi1s = second.phi.content[argsnested.content.content.i1]
    offsets_outer = argsnested.offsets
    offsets_inner = argsnested.content.offsets
    detas = np.abs(eta0s - eta1s)
    dphis = (phi0s - phi1s + math.pi) % (2 * math.pi) - math.pi
    passdr = ((detas**2 + dphis**2) < drCut2)
    passdr = awkward.JaggedArray.fromoffsets(offsets_inner, passdr)
    return awkward.JaggedArray.fromoffsets(offsets_outer, passdr.any())


class JaggedCandidateMethods(awkward.Methods):
    """
        JaggedCandidateMethods defines the additional methods that turn a JaggedArray
        into a JaggedCandidateArray suitable for most analysis work. Additional user-
        supplied attributes can be accessed via getattr or dot operators.
    """

    @classmethod
    def candidatesfromcounts(cls, counts, **kwargs):
        """
            cands = JaggedCandidateArray.candidatesfromcounts(counts=counts,
                                                              pt=column1,
                                                              eta=column2,
                                                              phi=column3,
                                                              mass=column4,
                                                              ...)
        """
        offsets = awkward.JaggedArray.counts2offsets(counts)
        return cls.candidatesfromoffsets(offsets, **kwargs)

    @classmethod
    def candidatesfromoffsets(cls, offsets, **kwargs):
        """
            cands = JaggedCandidateArray.candidatesfromoffsets(offsets=offsets,
                                                               pt=column1,
                                                               eta=column2,
                                                               phi=column3,
                                                               mass=column4,
                                                               ...)
        """
        items = kwargs
        argkeys = items.keys()
        p4 = None
        fast_pt = None
        fast_eta = None
        fast_phi = None
        fast_mass = None
        if 'p4' in argkeys:
            p4 = items['p4']
            if not isinstance(p4, uproot_methods.TLorentzVectorArray):
                p4 = uproot_methods.TLorentzVectorArray.from_cartesian(p4[:, 0], p4[:, 1],
                                                                       p4[:, 2], p4[:, 3])
            fast_pt = _fast_pt(p4)
            fast_eta = _fast_eta(p4)
            fast_phi = _fast_phi(p4)
            fast_mass = _fast_mass(p4)
        elif 'pt' in argkeys and 'eta' in argkeys and 'phi' in argkeys and 'mass' in argkeys:
            p4 = uproot_methods.TLorentzVectorArray.from_ptetaphim(items['pt'], items['eta'],
                                                                   items['phi'], items['mass'])
            fast_pt = items['pt']
            fast_eta = items['eta']
            fast_phi = items['phi']
            fast_mass = items['mass']
            del items['pt']
            del items['eta']
            del items['phi']
            del items['mass']
        elif 'pt' in argkeys and 'eta' in argkeys and 'phi' in argkeys and 'energy' in argkeys:
            p4 = uproot_methods.TLorentzVectorArray.from_ptetaphi(items['pt'], items['eta'],
                                                                  items['phi'], items['energy'])
            fast_pt = items['pt']
            fast_eta = items['eta']
            fast_phi = items['phi']
            fast_mass = _fast_mass(p4)
            del items['pt']
            del items['eta']
            del items['phi']
            del items['energy']
        elif 'px' in argkeys and 'py' in argkeys and 'pz' in argkeys and 'mass' in argkeys:
            p4 = uproot_methods.TLorentzVectorArray.from_xyzm(items['px'], items['py'],
                                                              items['pz'], items['mass'])
            fast_pt = _fast_pt(p4)
            fast_eta = _fast_eta(p4)
            fast_phi = _fast_phi(p4)
            fast_mass = items['mass']
            del items['px']
            del items['py']
            del items['pz']
            del items['mass']
        elif 'pt' in argkeys and 'phi' in argkeys and 'pz' in argkeys and 'energy' in argkeys:
            p4 = uproot_methods.TLorentzVectorArray.from_cylindrical(items['pt'], items['phi'],
                                                                     items['pz'], items['energy'])
            fast_pt = items['pt']
            fast_eta = _fast_eta(p4)
            fast_phi = items['phi']
            fast_mass = _fast_mass(p4)
            del items['pt']
            del items['phi']
            del items['pz']
            del items['energy']
        elif 'px' in argkeys and 'py' in argkeys and 'pz' in argkeys and 'energy' in argkeys:
            p4 = uproot_methods.TLorentzVectorArray.from_cartesian(items['px'], items['py'],
                                                                   items['pz'], items['energy'])
            fast_pt = _fast_pt(p4)
            fast_eta = _fast_eta(p4)
            fast_phi = _fast_phi(p4)
            fast_mass = _fast_mass(p4)
            del items['px']
            del items['py']
            del items['pz']
            del items['energy']
        elif 'p' in argkeys and 'theta' in argkeys and 'phi' in argkeys and 'energy' in argkeys:
            p4 = uproot_methods.TLorentzVectorArray.from_spherical(items['p'], items['theta'],
                                                                   items['phi'], items['energy'])
            fast_pt = _fast_pt(p4)
            fast_eta = _fast_eta(p4)
            fast_phi = items['phi']
            fast_mass = _fast_mass(p4)
            del items['p']
            del items['theta']
            del items['phi']
            del items['energy']
        elif 'p3' in argkeys and 'energy' in argkeys:
            p4 = uproot_methods.TLorentzVectorArray.from_p3(items['p3'], items['energy'])
            fast_pt = _fast_pt(p4)
            fast_eta = _fast_eta(p4)
            fast_phi = _fast_phi(p4)
            fast_mass = _fast_mass(p4)
            del items['p3']
            del items['energy']
        else:
            raise Exception('No valid definition of four-momentum found to build JaggedCandidateArray')

        items['p4'] = p4
        items['__fast_pt'] = fast_pt
        items['__fast_eta'] = fast_eta
        items['__fast_phi'] = fast_phi
        items['__fast_mass'] = fast_mass
        return cls.fromoffsets(offsets, awkward.Table(items))

    @property
    def p4(self):
        """ return TLorentzVectorArray of candidates """
        return self['p4']

    @property
    def pt(self):
        """ fast-cache version of pt """
        return self['__fast_pt']

    @property
    def eta(self):
        """ fast-cache version of eta """
        return self['__fast_eta']

    @property
    def phi(self):
        """ fast-cache version of phi """
        return self['__fast_phi']

    @property
    def mass(self):
        """ fast-cache version of mass """
        return self['__fast_mass']

    @property
    def i0(self):
        """ forward i0 from base """
        if 'p4' in self['0'].columns:
            return self.fromjagged(self['0'])
        return self['0']

    @property
    def i1(self):
        """ forward i1 from base """
        if 'p4' in self['1'].columns:
            return self.fromjagged(self['1'])
        return self['1']

    @property
    def i2(self):
        """ forward i2 from base """
        if 'p4' in self['2'].columns:
            return self.fromjagged(self['2'])
        return self['2']

    @property
    def i3(self):
        """ forward i3 from base """
        if 'p4' in self['3'].columns:
            return self.fromjagged(self['3'])
        return self['3']

    @property
    def i4(self):
        """ forward i4 from base """
        if 'p4' in self['4'].columns:
            return self.fromjagged(self['4'])
        return self['4']

    @property
    def i5(self):
        """ forward i5 from base """
        if 'p4' in self['5'].columns:
            return self.fromjagged(self['5'])
        return self['5']

    @property
    def i6(self):
        """ forward i6 from base """
        if 'p4' in self['6'].columns:
            return self.fromjagged(self['6'])
        return self['6']

    @property
    def i7(self):
        """ forward i7 from base """
        if 'p4' in self['7'].columns:
            return self.fromjagged(self['7'])
        return self['7']

    @property
    def i8(self):
        """ forward i8 from base """
        if 'p4' in self['8'].columns:
            return self.fromjagged(self['8'])
        return self['8']

    @property
    def i9(self):
        """ forward i9 from base """
        if 'p4' in self['9'].columns:
            return self.fromjagged(self['9'])
        return self['9']

    def add_attributes(self, **kwargs):
        """
            cands.add_attributes( name1 = column1,
                                  name2 = column2,
                                  ... )
        """
        for key, item in kwargs.items():
            if isinstance(item, awkward.JaggedArray):
                self[key] = awkward.JaggedArray.fromoffsets(self.offsets, item.content)
            elif isinstance(item, np.ndarray):
                self[key] = awkward.JaggedArray.fromoffsets(self.offsets, item)

    def distincts(self, nested=False):
        """
            This method calls the distincts method of JaggedArray to get all unique
            pairs per-event contained in a JaggedCandidateArray.
            The resulting JaggedArray of that call is dressed with the jagged candidate
            array four-momentum and cached fast access pt/eta/phi/mass.
        """
        outs = super(JaggedCandidateMethods, self).distincts(nested)
        outs['p4'] = outs.i0['p4'] + outs.i1['p4']
        thep4 = outs['p4']
        outs['__fast_pt'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_pt(thep4.content))
        outs['__fast_eta'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_eta(thep4.content))
        outs['__fast_phi'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_phi(thep4.content))
        outs['__fast_mass'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_mass(thep4.content))
        return self.fromjagged(outs)

    def pairs(self, nested=False):
        """
            This method calls the pairs method of JaggedArray to get all pairs
            per-event contained in a JaggedCandidateArray.
            The resulting JaggedArray of that call is dressed with the jagged candidate
            array four-momentum and cached fast access pt/eta/phi/mass.
        """
        outs = super(JaggedCandidateMethods, self).pairs(nested)
        outs['p4'] = outs.i0['p4'] + outs.i1['p4']
        thep4 = outs['p4']
        outs['__fast_pt'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_pt(thep4.content))
        outs['__fast_eta'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_eta(thep4.content))
        outs['__fast_phi'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_phi(thep4.content))
        outs['__fast_mass'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_mass(thep4.content))
        return self.fromjagged(outs)

    def cross(self, other, nested=False):
        """
            This method calls the cross method of JaggedArray to get all pairs
            per-event with another JaggedCandidateArray.
            The resulting JaggedArray of that call is dressed with the jagged candidate
            array four-momentum and cached fast access pt/eta/phi/mass.
        """
        outs = super(JaggedCandidateMethods, self).cross(other, nested)
        # currently JaggedArray.cross() has some funny behavior when it encounters the
        # p4 column and makes some wierd new column... for now I just delete it and reorder
        # everything looks ok after that
        keys = outs.columns
        reorder = False
        for key in keys:
            if not isinstance(outs[key].content, awkward.Table):
                del outs[key]
                reorder = True
        if reorder:
            keys = outs.columns
            realkey = {}
            for i in range(len(keys)):
                realkey[keys[i]] = str(i)
            for key in keys:
                if realkey[key] != key:
                    outs[realkey[key]] = outs[key]
                    del outs[key]
        keys = outs.columns
        for key in keys:
            if 'p4' not in outs.columns:
                outs['p4'] = outs[key]['p4']
            else:
                outs['p4'] = outs['p4'] + outs[key]['p4']
        thep4 = outs['p4']
        outs['__fast_pt'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_pt(thep4.content))
        outs['__fast_eta'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_eta(thep4.content))
        outs['__fast_phi'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_phi(thep4.content))
        outs['__fast_mass'] = awkward.JaggedArray.fromoffsets(outs.offsets, _fast_mass(thep4.content))
        return self.fromjagged(outs)

    # Function returns a mask with true or false at each location for whether object 1 matched with any object 2s
    # Optional parameter to add a cut on the percent pt difference between the objects
    def _matchcombs(self, cands):
        """
            Wrapper function that returns all p4 combinations of this JaggedCandidateArray
            with another input JaggedCandidateArray.
        """
        combinations = self.p4.cross(cands.p4, nested=True)
        if ((~(combinations.i0.pt > 0).flatten().flatten().all()) | (~(combinations.i1.pt > 0).flatten().flatten().all())):
            raise Exception("At least one particle has pt = 0")
        return combinations

    def match(self, cands, matchfunc=_default_match, **kwargs):
        """ returns a mask of candidates that pass matchfunc() """
        combinations = self._matchcombs(cands)
        return matchfunc(combinations, **kwargs)

    def fastmatch(self, cands, matchfunc=_default_fastmatch, **kwargs):
        return matchfunc(self, cands, **kwargs)

    # Function returns a fancy indexing.
    # At each object 1 location is the index of object 2 that it matched best with
    # <<<<important>>>> selves without a match will get a -1 to preserve counts structure
    def argmatch(self, cands, argmatchfunc=_default_argmatch, **kwargs):
        """
            returns a jagged array of indices that pass argmatchfunc.
            if there is no match a -1 is used to preserve the shape of
            the array represented by self
        """
        combinations = self._matchcombs(cands)
        return argmatchfunc(combinations, **kwargs)

    def __getattr__(self, what):
        """
            extend get attr to allow access to columns,
            gracefully thunk down to base methods
        """
        if what in self.columns:
            return self[what]
        thewhat = getattr(super(JaggedCandidateMethods, self), what)
        if 'p4' in thewhat.columns:
            return self.fromjagged(thewhat)
        return thewhat


JaggedCandidateArray = awkward.Methods.mixin(JaggedCandidateMethods, awkward.JaggedArray)
