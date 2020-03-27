# -*- coding: utf-8 -*-
from __future__ import division
from ..util import numpy as np
import scipy.stats
import copy
import warnings
import numbers
from .hist_tools import SparseAxis, DenseAxis, overflow_behavior, Interval, StringBin

# Plotting is always terrible
# Let's try our best to follow matplotlib idioms
# https://matplotlib.org/tutorials/introductory/usage.html#coding-styles

_coverage1sd = scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1)


def poisson_interval(sumw, sumw2, coverage=_coverage1sd):
    """Frequentist coverage interval for Poisson-distributed observations

    Parameters
    ----------
        sumw : numpy.ndarray
            Sum of weights vector
        sumw2 : numpy.ndarray
            Sum weights squared vector
        coverage : float, optional
            Central coverage interval, defaults to 68%

    Calculates the so-called 'Garwood' interval,
    c.f. https://www.ine.pt/revstat/pdf/rs120203.pdf or
    http://ms.mcmaster.ca/peter/s743/poissonalpha.html
    For weighted data, this approximates the observed count by ``sumw**2/sumw2``, which
    effectively scales the unweighted poisson interval by the average weight.
    This may not be the optimal solution: see https://arxiv.org/pdf/1309.1287.pdf for a proper treatment.
    When a bin is zero, the scale of the nearest nonzero bin is substituted to scale the nominal upper bound.
    If all bins zero, a warning is generated and interval is set to ``sumw``.
    """
    scale = np.empty_like(sumw)
    scale[sumw != 0] = sumw2[sumw != 0] / sumw[sumw != 0]
    if np.sum(sumw == 0) > 0:
        missing = np.where(sumw == 0)
        available = np.nonzero(sumw)
        if len(available[0]) == 0:
            warnings.warn("All sumw are zero!  Cannot compute meaningful error bars", RuntimeWarning)
            return np.vstack([sumw, sumw])
        nearest = sum([np.subtract.outer(d, d0)**2 for d, d0 in zip(available, missing)]).argmin(axis=0)
        argnearest = tuple(dim[nearest] for dim in available)
        scale[missing] = scale[argnearest]
    counts = sumw / scale
    lo = scale * scipy.stats.chi2.ppf((1 - coverage) / 2, 2 * counts) / 2.
    hi = scale * scipy.stats.chi2.ppf((1 + coverage) / 2, 2 * (counts + 1)) / 2.
    interval = np.array([lo, hi])
    interval[interval == np.nan] = 0.  # chi2.ppf produces nan for counts=0
    return interval


def clopper_pearson_interval(num, denom, coverage=_coverage1sd):
    """Compute Clopper-Pearson coverage interval for a binomial distribution

    Parameters
    ----------
        num : numpy.ndarray
            Numerator, or number of successes, vectorized
        denom : numpy.ndarray
            Denominator or number of trials, vectorized
        coverage : float, optional
            Central coverage interval, defaults to 68%

    c.f. http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if np.any(num > denom):
        raise ValueError("Found numerator larger than denominator while calculating binomial uncertainty")
    lo = scipy.stats.beta.ppf((1 - coverage) / 2, num, denom - num + 1)
    hi = scipy.stats.beta.ppf((1 + coverage) / 2, num + 1, denom - num)
    interval = np.array([lo, hi])
    interval[:, num == 0.] = 0.
    interval[1, num == denom] = 1.
    return interval


def normal_interval(pw, tw, pw2, tw2, coverage=_coverage1sd):
    """Compute errors based on the expansion of pass/(pass + fail), possibly weighted

        Parameters
        ----------
        pw : numpy.ndarray
            Numerator, or number of (weighted) successes, vectorized
        tw : numpy.ndarray
            Denominator or number of (weighted) trials, vectorized
        pw2 : numpy.ndarray
            Numerator sum of weights squared, vectorized
        tw2 : numpy.ndarray
            Denominator sum of weights squared, vectorized
        coverage : float, optional
            Central coverage interval, defaults to 68%

        c.f. https://root.cern.ch/doc/master/TEfficiency_8cxx_source.html#l02515
    """

    eff = pw / tw

    variance = (pw2 * (1 - 2 * eff) + tw2 * eff**2) / (tw**2)
    sigma = np.sqrt(variance)

    prob = 0.5 * (1 - coverage)
    delta = np.zeros_like(sigma)
    delta[sigma != 0] = scipy.stats.norm.ppf(prob, scale=sigma[sigma != 0])

    lo = eff - np.minimum(eff + delta, np.ones_like(eff))
    hi = np.maximum(eff - delta, np.zeros_like(eff)) - eff

    return np.array([lo, hi])


def plot1d(hist, ax=None, clear=True, overlay=None, stack=False, overflow='none', line_opts=None,
           fill_opts=None, error_opts=None, legend_opts={}, overlay_overflow='none',
           density=False, binwnorm=None, densitymode='unit', order=None):
    """Create a 1D plot from a 1D or 2D `Hist` object

    Parameters
    ----------
        hist : Hist
            Histogram with maximum of two dimensions
        ax : matplotlib.axes.Axes, optional
            Axes object (if None, one is created)
        clear : bool, optional
            Whether to clear Axes before drawing (if passed); if False, this function will skip drawing the legend
        overlay : str, optional
            In the case that ``hist`` is 2D, specify the axis of hist to overlay (remaining axis will be x axis)
        stack : bool, optional
            Whether to stack or overlay non-axis dimension (if it exists)
        order : list, optional
            How to order when stacking. Take a list of identifiers.
        overflow : str, optional
            If overflow behavior is not 'none', extra bins will be drawn on either end of the nominal
            axis range, to represent the contents of the overflow bins.  See `Hist.sum` documentation
            for a description of the options.
        line_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.step <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.step.html>`_ call
            internal to this function.  Leave blank for defaults.
        fill_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.fill_between <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.fill_between.html>`_ call
            internal to this function.  Leave blank for defaults.
        error_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.errorbar <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html>`_ call
            internal to this function.  Leave blank for defaults.  Some special options are interpreted by
            this function and not passed to matplotlib: 'emarker' (default: '') specifies the marker type
            to place at cap of the errorbar.
        legend_opts : dict, optional
            A dictionary of options  to pass to the matplotlib
            `ax.legend <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.legend.html>`_ call
            internal to this fuction.  Leave blank for defaults.
        overlay_overflow : str, optional
            If overflow behavior is not 'none', extra bins in the overlay axis will be overlayed or stacked,
            to represent the contents of the overflow bins.  See `Hist.sum` documentation for a description of the options.
        density : bool, optional
            If true, convert sum weights to probability density (i.e. integrates to 1 over domain of axis)
            (Note: this option conflicts with ``binwnorm``)
        densitymode: ["unit", "stack"], default: "unit"
            If using both density/binwnorm and stack choose stacking behaviour. "unit" normalized
            each histogram separately and stacks afterwards, while "stack" normalizes the total after summing.
        binwnorm : float, optional
            If true, convert sum weights to bin-width-normalized, with unit equal to supplied value (usually you want to specify 1.)


    Returns
    -------
        ax : matplotlib.axes.Axes
            A matplotlib `Axes <https://matplotlib.org/3.1.1/api/axes_api.html>`_ object
    """
    import mplhep as hep
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
    if hist.dim() > 2:
        raise ValueError("plot1d() can only support up to two dimensions (one for axis, one to stack or overlay)")
    if overlay is None and hist.sparse_dim() == 1 and hist.dense_dim() == 1:
        overlay = hist.sparse_axes()[0].name
    elif overlay is None and hist.dim() > 1:
        raise ValueError("plot1d() can only support one dimension without an overlay axis chosen")
    if density and binwnorm is not None:
        raise ValueError("Cannot use density and binwnorm at the same time!")
    if binwnorm is not None:
        if not isinstance(binwnorm, numbers.Number):
            raise ValueError("Bin width normalization not a number, but a %r" % binwnorm.__class__)
    if line_opts is None and fill_opts is None and error_opts is None:
        if stack:
            fill_opts = {}
        else:
            line_opts = {}
            error_opts = {}

    axis = hist.axes()[0]
    if overlay is not None:
        overlay = hist.axis(overlay)
        if axis == overlay:
            axis = hist.axes()[1]
    if isinstance(axis, SparseAxis):
        raise NotImplementedError("Plot a sparse axis (e.g. bar chart)")
    elif isinstance(axis, DenseAxis):
        ax.set_xlabel(axis.label)
        ax.set_ylabel(hist.label)
        edges = axis.edges(overflow=overflow)
        if order is None:
            identifiers = hist.identifiers(overlay, overflow=overlay_overflow) if overlay is not None else [None]
        else:
            identifiers = order
        plot_info = {
            'identifier': identifiers,
            'label': list(map(str, identifiers)),
            'sumw': [],
            'sumw2': []
        }
        for i, identifier in enumerate(identifiers):
            if identifier is None:
                sumw, sumw2 = hist.values(sumw2=True, overflow=overflow)[()]
            elif isinstance(overlay, SparseAxis):
                sumw, sumw2 = hist.integrate(overlay, identifier).values(sumw2=True, overflow=overflow)[()]
            else:
                sumw, sumw2 = hist.values(sumw2=True, overflow='allnan')[()]
                the_slice = (i if overflow_behavior(overlay_overflow).start is None else i + 1, overflow_behavior(overflow))
                if hist._idense(overlay) == 1:
                    the_slice = (the_slice[1], the_slice[0])
                sumw = sumw[the_slice]
                sumw2 = sumw2[the_slice]
            plot_info['sumw'].append(sumw)
            plot_info['sumw2'].append(sumw2)

        def w2err(sumw, sumw2):
            err = []
            for a, b in zip(sumw, sumw2):
                err.append(np.abs(poisson_interval(a, b) - a))
            return err

        kwargs = None
        if line_opts is not None and error_opts is None:
            _error = None
        else:
            _error = w2err(plot_info['sumw'], plot_info['sumw2'])
        if fill_opts is not None:
            histtype = 'fill'
            kwargs = fill_opts
        elif error_opts is not None and line_opts is None:
            histtype = 'errorbar'
            kwargs = error_opts
        else:
            histtype = 'step'
            kwargs = line_opts
        if kwargs is None:
            kwargs = {}

        hep.histplot(plot_info['sumw'], edges, label=plot_info['label'],
                     yerr=_error, histtype=histtype, ax=ax,
                     density=density, binwnorm=binwnorm, stack=stack,
                     densitymode=densitymode,
                     **kwargs)

        if stack and error_opts is not None:
            stack_sumw = np.sum(plot_info['sumw'], axis=0)
            stack_sumw2 = np.sum(plot_info['sumw2'], axis=0)
            err = poisson_interval(stack_sumw, stack_sumw2)
            opts = {'step': 'post', 'label': 'Sum unc.', 'hatch': '///',
                    'facecolor': 'none', 'edgecolor': (0, 0, 0, .5), 'linewidth': 0}
            opts.update(error_opts)
            ax.fill_between(x=edges, y1=np.r_[err[0, :], err[0, -1]],
                            y2=np.r_[err[1, :], err[1, -1]], **opts)

        if legend_opts is not None:
            _label = overlay.label if overlay is not None else ""
            ax.legend(title=_label, **legend_opts)
        else:
            ax.legend(title=_label)
        ax.autoscale(axis='x', tight=True)
        ax.set_ylim(0, None)

    return ax


def plotratio(num, denom, ax=None, clear=True, overflow='none', error_opts=None, denom_fill_opts=None, guide_opts=None, unc='clopper-pearson', label=None):
    """Create a ratio plot, dividing two compatible histograms

    Parameters
    ----------
        num : Hist
            Numerator, a single-axis histogram
        denom : Hist
            Denominator, a single-axis histogram
        ax : matplotlib.axes.Axes, optional
            Axes object (if None, one is created)
        clear : bool, optional
            Whether to clear Axes before drawing (if passed); if False, this function will skip drawing the legend
        overflow : str, optional
            If overflow behavior is not 'none', extra bins will be drawn on either end of the nominal
            axis range, to represent the contents of the overflow bins.  See `Hist.sum` documentation
            for a description of the options.
        error_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.errorbar <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.errorbar.html>`_ call
            internal to this function.  Leave blank for defaults.  Some special options are interpreted by
            this function and not passed to matplotlib: 'emarker' (default: '') specifies the marker type
            to place at cap of the errorbar.
        denom_fill_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.fill_between <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.fill_between.html>`_ call
            internal to this function, filling the denominator uncertainty band.  Leave blank for defaults.
        guide_opts : dict, optional
            A dictionary of options to pass to the matplotlib
            `ax.axhline <https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.axhline.html>`_ call
            internal to this function, to plot a horizontal guide line at ratio of 1.  Leave blank for defaults.
        unc : str, optional
            Uncertainty calculation option: 'clopper-pearson' interval for efficiencies; 'poisson-ratio' interval
            for ratio of poisson distributions; 'num' poisson interval of numerator scaled by denominator value
            (common for data/mc, for better or worse).
        label : str, optional
            Associate a label to this entry (note: y axis label set by ``num.label``)

    Returns
    -------
        ax : matplotlib.axes.Axes
            A matplotlib `Axes <https://matplotlib.org/3.1.1/api/axes_api.html>`_ object
    """
    import mplhep as hep
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
    if not num.compatible(denom):
        raise ValueError("numerator and denominator histograms have incompatible axis definitions")
    if num.dim() > 1:
        raise ValueError("plotratio() can only support one-dimensional histograms")
    if error_opts is None and denom_fill_opts is None and guide_opts is None:
        error_opts = {}
        denom_fill_opts = {}

    axis = num.axes()[0]
    if isinstance(axis, SparseAxis):
        raise NotImplementedError("Ratio for sparse axes (labeled axis with errorbars)")
    elif isinstance(axis, DenseAxis):
        ax.set_xlabel(axis.label)
        ax.set_ylabel(num.label)
        edges = axis.edges(overflow=overflow)
        centers = axis.centers(overflow=overflow)

        sumw_num, sumw2_num = num.values(sumw2=True, overflow=overflow)[()]
        sumw_denom, sumw2_denom = denom.values(sumw2=True, overflow=overflow)[()]

        rsumw = sumw_num / sumw_denom
        if unc == 'clopper-pearson':
            rsumw_err = np.abs(clopper_pearson_interval(sumw_num, sumw_denom) - rsumw)
        elif unc == 'poisson-ratio':
            # poisson ratio n/m is equivalent to binomial n/(n+m)
            rsumw_err = np.abs(clopper_pearson_interval(sumw_num, sumw_num + sumw_denom) - rsumw)
        elif unc == 'num':
            rsumw_err = np.abs(poisson_interval(rsumw, sumw2_num / sumw_denom**2) - rsumw)
        elif unc == "normal":
            rsumw_err = np.abs(normal_interval(sumw_num, sumw_denom, sumw2_num, sumw2_denom))
        else:
            raise ValueError("Unrecognized uncertainty option: %r" % unc)

        if error_opts is not None:
            opts = {'label': label, 'linestyle': 'none'}
            opts.update(error_opts)
            emarker = opts.pop('emarker', '')
            errbar = ax.errorbar(x=centers, y=rsumw, yerr=rsumw_err, **opts)
            plt.setp(errbar[1], 'marker', emarker)
        if denom_fill_opts is not None:
            unity = np.ones_like(sumw_denom)
            denom_unc = poisson_interval(unity, sumw2_denom / sumw_denom**2)
            opts = {'step': 'post', 'facecolor': (0, 0, 0, 0.3), 'linewidth': 0}
            opts.update(denom_fill_opts)
            ax.fill_between(edges, np.r_[denom_unc[0], denom_unc[0, -1]], np.r_[denom_unc[1], denom_unc[1, -1]], **opts)
        if guide_opts is not None:
            opts = {'linestyle': '--', 'color': (0, 0, 0, 0.5), 'linewidth': 1}
            opts.update(guide_opts)
            ax.axhline(1., **opts)

    if clear:
        ax.autoscale(axis='x', tight=True)
        ax.set_ylim(0, None)

    return ax


def plot2d(hist, xaxis, ax=None, clear=True, xoverflow='none', yoverflow='none', patch_opts=None, text_opts=None, density=False, binwnorm=None):
    """Create a 2D plot from a 2D `Hist` object

    Parameters
    ----------
        hist : Hist
            Histogram with two dimensions
        xaxis : str or Axis
            Which of the two dimensions to use as an x axis
        ax : matplotlib.axes.Axes, optional
            Axes object (if None, one is created)
        clear : bool, optional
            Whether to clear Axes before drawing (if passed); if False, this function will skip drawing the legend
        xoverflow : str, optional
            If overflow behavior is not 'none', extra bins will be drawn on either end of the nominal x
            axis range, to represent the contents of the overflow bins.  See `Hist.sum` documentation
            for a description of the options.
        yoverflow : str, optional
            Similar to ``xoverflow``
        patch_opts : dict, optional
            Options passed to the matplotlib `pcolormesh <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolormesh.html>`_
            call internal to this function, to plot a rectangular grid of patches colored according to the bin values.
            Leave empty for defaults.
        text_opts : dict, optional
            Options passed to the matplotlib `text <https://matplotlib.org/api/text_api.html#matplotlib.text.Text>`_
            call internal to this function, to place a text label at each bin center with the bin value.  Special
            options interpreted by this function and not passed to matplotlib: 'format': printf-style float format
            , default '%.2g'.
        density : bool, optional
            If true, convert sum weights to probability density (i.e. integrates to 1 over domain of axis)
            (Note: this option conflicts with ``binwnorm``)
        binwnorm : float, optional
            If true, convert sum weights to bin-width-normalized, with unit equal to supplied value (usually you want to specify 1.)

    Returns
    -------
        ax : matplotlib.axes.Axes
            A matplotlib `Axes <https://matplotlib.org/3.1.1/api/axes_api.html>`_ object
    """
    import mplhep as hep
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
        fig = ax.figure
    if hist.dim() != 2:
        raise ValueError("plot2d() can only support exactly two dimensions")
    if density and binwnorm is not None:
        raise ValueError("Cannot use density and binwnorm at the same time!")
    if binwnorm is not None:
        if not isinstance(binwnorm, numbers.Number):
            raise ValueError("Bin width normalization not a number, but a %r" % binwnorm.__class__)
    if patch_opts is None and text_opts is None:
        patch_opts = {}

    xaxis = hist.axis(xaxis)
    yaxis = hist.axes()[1]
    transpose = False
    if yaxis == xaxis:
        yaxis = hist.axes()[0]
        transpose = True
    if isinstance(xaxis, SparseAxis) or isinstance(yaxis, SparseAxis):
        raise NotImplementedError("Plot a sparse axis (e.g. bar chart or labeled bins)")
    else:
        xedges = xaxis.edges(overflow=xoverflow)
        yedges = yaxis.edges(overflow=yoverflow)
        sumw, sumw2 = hist.values(sumw2=True, overflow='allnan')[()]
        if transpose:
            sumw = sumw.T
            sumw2 = sumw2.T
        # no support for different overflow behavior per axis, do it ourselves
        sumw = sumw[overflow_behavior(xoverflow), overflow_behavior(yoverflow)]
        sumw2 = sumw2[overflow_behavior(xoverflow), overflow_behavior(yoverflow)]
        if (density or binwnorm is not None) and np.sum(sumw) > 0:
            overallnorm = np.sum(sumw) * binwnorm if binwnorm is not None else 1.
            areas = np.multiply.outer(np.diff(xedges), np.diff(yedges))
            binnorms = overallnorm / (areas * np.sum(sumw))
            sumw = sumw * binnorms
            sumw2 = sumw2 * binnorms**2

        if patch_opts is not None:
            opts = {'cmap': 'viridis'}
            opts.update(patch_opts)
            pc = ax.pcolormesh(xedges, yedges, sumw.T, **opts)
            ax.add_collection(pc)
            if clear:
                fig.colorbar(pc, ax=ax, label=hist.label)
        if text_opts is not None:
            for ix, xcenter in enumerate(xaxis.centers()):
                for iy, ycenter in enumerate(yaxis.centers()):
                    opts = {
                        'horizontalalignment': 'center',
                        'verticalalignment': 'center',
                    }
                    if patch_opts is not None:
                        opts['color'] = 'black' if pc.norm(sumw[ix, iy]) > 0.5 else 'lightgrey'
                    opts.update(text_opts)
                    txtformat = opts.pop('format', r'%.2g')
                    ax.text(xcenter, ycenter, txtformat % sumw[ix, iy], **opts)

    if clear:
        ax.set_xlabel(xaxis.label)
        ax.set_ylabel(yaxis.label)
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])

    return ax


def plotgrid(h, figure=None, row=None, col=None, overlay=None, row_overflow='none', col_overflow='none', **plot_opts):
    """Create a grid of plots, enumerating identifiers on up to 3 axes

    Parameters
    ----------
        h : Hist
            A histogram with up to 3 axes
        figure : matplotlib.figure.Figure, optional
            If omitted, a new figure is created. Otherwise, the axes will be redrawn on this existing figure.
        row : str
            Name of row axis
        col : str
            Name of column axis
        overlay : str
            name of overlay axis
        row_overflow : str, optional
            If overflow behavior is not 'none', extra bins will be drawn on either end of the nominal x
            axis range, to represent the contents of the overflow bins.  See `Hist.sum` documentation
            for a description of the options.
        col_overflow : str, optional
            Similar to ``row_overflow``
        ``**plot_opts`` : kwargs
            The remaining axis of the histogram, after removing any of ``row,col,overlay`` specified,
            will be the plot axis, with ``plot_opts`` passed to the `plot1d` call.

    Returns
    -------
        axes : numpy.ndarray
            An array of matplotlib `Axes <https://matplotlib.org/3.1.1/api/axes_api.html>`_ objects
    """
    import mplhep as hep
    import matplotlib.pyplot as plt
    haxes = set(ax.name for ax in h.axes())
    nrow, ncol = 1, 1
    if row:
        row_identifiers = h.identifiers(row, overflow=row_overflow)
        nrow = len(row_identifiers)
        haxes.remove(row)
    if col:
        col_identifiers = h.identifiers(col, overflow=col_overflow)
        ncol = len(col_identifiers)
        haxes.remove(col)
    if overlay:
        haxes.remove(overlay)
    if len(haxes) > 1:
        raise ValueError("More than one dimension left: %s" % (",".join(ax for ax in haxes),))
    elif len(haxes) == 0:
        raise ValueError("Not enough dimensions available in %r" % h)

    figsize = plt.rcParams['figure.figsize']
    figsize = figsize[0] * max(ncol, 1), figsize[1] * max(nrow, 1)
    if figure is None:
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    else:
        fig = figure
        shape = (0, 0)
        lastax = fig.get_children()[-1]
        if isinstance(lastax, plt.Axes):
            shape = lastax.rowNum + 1, lastax.colNum + 1
        if shape[0] == nrow and shape[1] == ncol:
            axes = np.array(fig.axes).reshape(shape)
        else:
            fig.clear()
            # fig.set_size_inches(figsize)
            axes = fig.subplots(nrow, ncol, squeeze=False, sharex=True, sharey=True)

    for icol in range(ncol):
        hcol = h
        coltitle = None
        if col:
            vcol = col_identifiers[icol]
            hcol = h.integrate(col, vcol)
            coltitle = str(vcol)
            if isinstance(vcol, Interval) and vcol.label is None:
                coltitle = "%s ∈ %s" % (h.axis(col).label, coltitle)
        for irow in range(nrow):
            ax = axes[irow, icol]
            hplot = hcol
            rowtitle = None
            if row:
                vrow = row_identifiers[irow]
                hplot = hcol.integrate(row, vrow)
                rowtitle = str(vrow)
                if isinstance(vrow, Interval) and vrow.label is None:
                    rowtitle = "%s ∈ %s" % (h.axis(row).label, rowtitle)

            plot1d(hplot, ax=ax, overlay=overlay, **plot_opts)
            if row is not None and col is not None:
                ax.set_title("%s, %s" % (rowtitle, coltitle))
            elif row is not None:
                ax.set_title(rowtitle)
            elif col is not None:
                ax.set_title(coltitle)

    for ax in axes.flatten():
        ax.autoscale(axis='y')
        ax.set_ylim(0, None)

    return axes
