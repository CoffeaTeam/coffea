import awkward

from ..util import awkward as util_awkward
import uproot_methods
from uproot_methods.classes.TLorentzVector import ArrayMethods, TLorentzVectorArray, TLorentzVector

JaggedArrayMethods = ArrayMethods.mixin(ArrayMethods, util_awkward.JaggedArray)


class Dangerousness(object):
    # NB: This is dangerous
    def __getattr__(self, what):
        if what in self.columns:
            return self[what]
        thewhat = getattr(super(Dangerousness, self), what)
        return self.fromjagged(thewhat)


class SaiyanCommonMethods(Dangerousness, ArrayMethods):
    def match(self, cands, value):
        if isinstance(self, awkward.JaggedArray):
            combinations = self.cross(cands, nested=True)
            mask = (combinations.i0.delta_r(combinations.i1) < value)
        else:
            mask = self.delta_r(cands) < value
        return mask.any()

    def closest(self, cands):
        if isinstance(self, awkward.JaggedArray):
            combinations = self.cross(cands)
            if ((~(combinations.i0.eta == 0).flatten().flatten().all()) | (~(combinations.i1.eta == 0).flatten().flatten().all())):
                criterium = combinations.i0.delta_phi(combinations.i1)
            else:
                criterium = combinations.i0.delta_r(combinations.i1)
        else:
            if ((~(self.eta == 0).flatten().flatten().all()) | (~(cands.eta == 0).flatten().flatten().all())):
                criterium = self.delta_phi(cands)
            else:
                criterium = self.delta_r(cands)
        return cands[criterium.argmin()]


class JaggedSaiyanArrayMethods(SaiyanCommonMethods, JaggedArrayMethods):
    def pairs(self, nested=False):
        outs = super(JaggedSaiyanArrayMethods, self).pairs(nested)
        return self.fromjagged(outs)

    def distincts(self, nested=False):
        outs = super(JaggedSaiyanArrayMethods, self).distincts(nested)
        return self.fromjagged(outs)

    def cross(self, other, nested=False):
        outs = super(JaggedSaiyanArrayMethods, self).cross(other, nested)
        return self.fromjagged(outs)


class SaiyanArray(SaiyanCommonMethods, TLorentzVectorArray):
    def _initObjectArray(self, table):
        self.awkward.ObjectArray.__init__(self, table, lambda row: TLorentzVector(row["fX"], row["fY"], row["fZ"], row["fE"]))


def Initialize(items):
    argkeys = items.keys()
    p4 = None

    if 'p4' in argkeys:
        p4 = items['p4']
        if not isinstance(p4, uproot_methods.TLorentzVectorArray):
            p4 = SaiyanArray.from_cartesian(p4[:, 0], p4[:, 1],
                                            p4[:, 2], p4[:, 3])

    elif 'pt' in argkeys and 'eta' in argkeys and 'phi' in argkeys and 'mass' in argkeys:
        temp = SaiyanArray.from_ptetaphim(items['pt'], items['eta'],
                                          items['phi'], items['mass'])._to_cartesian()
        p4 = SaiyanArray.from_cartesian(temp.x, temp.y, temp.z, temp.t)
        del items['pt']
        del items['eta']
        del items['phi']
        del items['mass']

    elif 'pt' in argkeys and 'eta' in argkeys and 'phi' in argkeys and 'energy' in argkeys:
        p4 = SaiyanArray.from_ptetaphi(items['pt'], items['eta'],
                                       items['phi'], items['energy'])
        del items['pt']
        del items['eta']
        del items['phi']
        del items['energy']

    elif 'px' in argkeys and 'py' in argkeys and 'pz' in argkeys and 'mass' in argkeys:
        p4 = SaiyanArray.from_xyzm(items['px'], items['py'],
                                   items['pz'], items['mass'])
        del items['px']
        del items['py']
        del items['pz']
        del items['mass']

    elif 'pt' in argkeys and 'phi' in argkeys and 'pz' in argkeys and 'energy' in argkeys:
        p4 = SaiyanArray.from_cylindrical(items['pt'], items['phi'],
                                          items['pz'], items['energy'])
        del items['pt']
        del items['phi']
        del items['pz']
        del items['energy']

    elif 'px' in argkeys and 'py' in argkeys and 'pz' in argkeys and 'energy' in argkeys:
        p4 = SaiyanArray.from_cartesian(items['px'], items['py'],
                                        items['pz'], items['energy'])
        del items['px']
        del items['py']
        del items['pz']
        del items['energy']

    elif 'p' in argkeys and 'theta' in argkeys and 'phi' in argkeys and 'energy' in argkeys:
        p4 = SaiyanArray.from_spherical(items['p'], items['theta'],
                                        items['phi'], items['energy'])
        del items['p']
        del items['theta']
        del items['phi']
        del items['energy']

    elif 'p3' in argkeys and 'energy' in argkeys:
        p4 = SaiyanArray.from_p3(items['p3'], items['energy'])
        del items['p3']
        del items['energy']

    try:
        p4
    except NameError:
        out = awkward.Table()
    else:
        out = p4

    for name, value in items.items():
        out[name] = value

    if isinstance(out, awkward.JaggedArray):
        out = JaggedSaiyanArrayMethods.fromjagged(out)
    else:  # p4 is not None and not isinstance(out, awkward.JaggedArray):
        try:
            p4
        except NameError:
            out.__class__ = type("Event", (out.__class__,
                                           Dangerousness), {})
        else:
            out.__class__ = type("Object", (out.__class__,
                                            SaiyanCommonMethods), {})

    return out
