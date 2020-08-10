from .hist_tools import SparseAxis, DenseAxis
from uproot_methods.classes.TH1 import Methods as TH1Methods


class TH1(TH1Methods, list):
    pass


class TAxis(object):
    def __init__(self, fNbins, fXmin, fXmax):
        self._fNbins = fNbins
        self._fXmin = fXmin
        self._fXmax = fXmax


def export1d(hist):
    """Export a 1-dimensional `Hist` object to uproot

    This allows one to write a coffea histogram into a ROOT file, via uproot.

    Parameters
    ----------
        hist : Hist
            A 1-dimensional histogram object

    Returns
    -------
        out
            A ``uproot_methods.classes.TH1`` object

    Examples
    --------
    Creating a coffea histogram, filling, and writing to a file::

        import coffea, uproot, numpy
        h = coffea.hist.Hist("Events", coffea.hist.Bin("var", "some variable", 20, 0, 1))
        h.fill(var=numpy.random.normal(size=100))
        fout = uproot.create('output.root')
        fout['myhist'] = coffea.hist.export1d(h)
        fout.close()

    """
    if hist.dense_dim() != 1:
        raise ValueError("export1d() can only support one dense dimension")
    if hist.sparse_dim() != 0:
        raise ValueError("export1d() expects zero sparse dimensions")

    axis = hist.axes()[0]
    sumw, sumw2 = hist.values(sumw2=True, overflow='all')[()]
    edges = axis.edges(overflow='none')

    out = TH1.__new__(TH1)
    out._fXaxis = TAxis(len(edges) - 1, edges[0], edges[-1])
    out._fXaxis._fName = axis.name
    out._fXaxis._fTitle = axis.label
    if not axis._uniform:
        out._fXaxis._fXbins = edges.astype(">f8")

    centers = (edges[:-1] + edges[1:]) / 2.0
    out._fEntries = out._fTsumw = out._fTsumw2 = sumw[1:-1].sum()
    out._fTsumwx = (sumw[1:-1] * centers).sum()
    out._fTsumwx2 = (sumw[1:-1] * centers**2).sum()

    out._fName = "histogram"
    out._fTitle = hist.label

    out._classname = b"TH1D"
    out.extend(sumw.astype(">f8"))
    out._fSumw2 = sumw2.astype(">f8")

    return out
