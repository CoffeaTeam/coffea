# -*- coding: utf-8 -*-
from __future__ import division
from fnal_column_analysis_tools.util import numpy as np
import scipy.stats
import copy
import warnings
import numbers
from .hist_tools import SparseAxis, DenseAxis, overflow_behavior, Interval, StringBin

import matplotlib.pyplot as plt

# Plotting is always terrible
# Let's try our best to follow matplotlib idioms
# https://matplotlib.org/tutorials/introductory/usage.html#coding-styles

_coverage1sd = scipy.stats.norm.cdf(1) - scipy.stats.norm.cdf(-1)


def poisson_interval(sumw, sumw2, coverage=_coverage1sd):
    """
        sumw: sum of weights
        sumw2: sum weights**2
        coverage: coverage, default to 68%

        The so-called 'Garwood' interval
            c.f. https://www.ine.pt/revstat/pdf/rs120203.pdf
            or http://ms.mcmaster.ca/peter/s743/poissonalpha.html
        For weighted data, approximate the observed count by sumw**2/sumw2
            This choice effectively scales the unweighted poisson interval by the average weight
            Maybe not the best... see https://arxiv.org/pdf/1309.1287.pdf for a proper treatment
        When a bin is zero, find the scale of the nearest nonzero bin
        If all bins zero, raise warning and set interval to sumw
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
    """
        Compute Clopper-Pearson coverage interval for binomial distribution
        num: successes
        denom: trials
        coverage: coverage, default to 68%

        c.f. http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    """
    if np.any(num > denom):
        raise ValueError("Found numerator larger than denominator while calculating binomial uncertainty")
    lo = scipy.stats.beta.ppf((1 - coverage) / 2, num, denom - num + 1)
    hi = scipy.stats.beta.ppf((1 + coverage) / 2, num + 1, denom - num)
    interval = np.array([lo, hi])
    interval[:, num == 0.] = 0.
    interval[:, num == denom] = 1.
    return interval


def plot1d(hist, ax=None, clear=True, overlay=None, stack=False, overflow='none', line_opts=None,
           fill_opts=None, error_opts=None, overlay_overflow='none', density=False, binwnorm=None):
    """
        hist: Hist object with maximum of two dimensions
        ax: matplotlib Axes object (if None, one is created)
        clear: clear Axes before drawing (if passed); if False, this function will skip drawing the legend
        overlay: the axis of hist to overlay (remaining one will be x axis)
        stack: whether to stack or overlay the other dimension (if one exists)
        overflow: overflow behavior of plot axis (see Hist.sum() docs)

        The draw options are passed as dicts to the relevant matplotlib function, with some exceptions in case
        it is especially common or useful.  If none of *_opts is specified, nothing will be plotted!
        Pass an empty dict (e.g. line_opts={}) for defaults
            line_opts: options to plot a step
                Special options interpreted by this function and not passed to matplotlib:
                    (none)

            fill_opts: to plot a filled area
                Special options interpreted by this function and not passed to matplotlib:
                    (none)

            error_opts: to plot an errorbar
                Special options interpreted by this function and not passed to matplotlib:
                    'emarker' (default: '') marker to place at cap of errorbar


        overlay_overflow: overflow behavior of dense overlay axis, if one exists
        density: Convert sum weights to probability density (i.e. integrates to 1 over domain of axis) (NB: conflicts with binwnorm)
        binwnorm: Convert sum weights to bin-width-normalized, with units equal to supplied value (usually you want to specify 1.)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
        fig = ax.figure
    if hist.dim() > 2:
        raise ValueError("plot1d() can only support up to two dimensions (one for axis, one to stack or overlay)")
    if overlay is None and hist.dim() > 1:
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
        centers = axis.centers(overflow=overflow)
        stack_sumw, stack_sumw2 = None, None
        primitives = {}
        identifiers = hist.identifiers(overlay, overflow=overlay_overflow) if overlay is not None else [None]
        for i, identifier in enumerate(identifiers):
            if identifier is None:
                sumw, sumw2 = hist.values(sumw2=True, overflow=overflow)[()]
            elif isinstance(overlay, SparseAxis):
                sumw, sumw2 = hist.project(overlay, identifier).values(sumw2=True, overflow=overflow)[()]
            else:
                sumw, sumw2 = hist.values(sumw2=True, overflow='allnan')[()]
                the_slice = (i if overflow_behavior(overlay_overflow).start is None else i + 1, overflow_behavior(overflow))
                if hist._idense(overlay) == 1:
                    the_slice = (the_slice[1], the_slice[0])
                sumw = sumw[the_slice]
                sumw2 = sumw2[the_slice]
            if (density or binwnorm is not None) and np.sum(sumw) > 0:
                overallnorm = np.sum(sumw) * binwnorm if binwnorm is not None else 1.
                binnorms = overallnorm / (np.diff(edges) * np.sum(sumw))
                sumw = sumw * binnorms
                sumw2 = sumw2 * binnorms**2
            label = str(identifier)
            primitives[identifier] = []
            first_color = None
            if stack:
                if stack_sumw is None:
                    stack_sumw, stack_sumw2 = sumw.copy(), sumw2.copy()
                else:
                    stack_sumw += sumw
                    stack_sumw2 += sumw2

                if line_opts is not None:
                    opts = {'where': 'post', 'label': label}
                    opts.update(line_opts)
                    l, = ax.step(x=edges, y=np.r_[stack_sumw, stack_sumw[-1]], **opts)
                    first_color = l.get_color()
                    primitives[identifier].append(l)
                if fill_opts is not None:
                    opts = {'step': 'post', 'label': label}
                    if first_color is not None:
                        opts['color'] = first_color
                    opts.update(fill_opts)
                    f = ax.fill_between(x=edges, y1=np.r_[stack_sumw - sumw, stack_sumw[-1] - sumw[-1]], y2=np.r_[stack_sumw, stack_sumw[-1]], **opts)
                    if first_color is None:
                        first_color = f.get_facecolor()[0]
                    primitives[identifier].append(f)
                # error_opts for stack is interpreted later
            else:
                if line_opts is not None:
                    opts = {'where': 'post', 'label': label}
                    opts.update(line_opts)
                    l, = ax.step(x=edges, y=np.r_[sumw, sumw[-1]], **opts)
                    first_color = l.get_color()
                    primitives[identifier].append(l)
                if fill_opts is not None:
                    opts = {'step': 'post', 'label': label}
                    if first_color is not None:
                        opts['color'] = first_color
                    opts.update(fill_opts)
                    f = ax.fill_between(x=edges, y1=np.r_[sumw, sumw[-1]], **opts)
                    if first_color is None:
                        first_color = f.get_facecolor()[0]
                    primitives[identifier].append(f)
                if error_opts is not None:
                    opts = {'linestyle': 'none', 'label': label}
                    if first_color is not None:
                        opts['color'] = first_color
                    opts.update(error_opts)
                    emarker = opts.pop('emarker', '')
                    err = np.abs(poisson_interval(sumw, sumw2) - sumw)
                    errbar = ax.errorbar(x=centers, y=sumw, yerr=err, **opts)
                    plt.setp(errbar[1], 'marker', emarker)
                    primitives[identifier].append(errbar)
        if stack_sumw is not None and error_opts is not None:
            err = poisson_interval(stack_sumw, stack_sumw2)
            opts = {'step': 'post', 'label': 'Sum unc.'}
            opts.update(error_opts)
            errbar = ax.fill_between(x=edges, y1=np.r_[err[0, :], err[0, -1]], y2=np.r_[err[1, :], err[1, -1]], **opts)
            primitives[StringBin('stack_unc', opts['label'])] = [errbar]

    if clear:
        if overlay is not None:
            handles, labels = list(), list()
            for identifier, handlelist in primitives.items():
                handles.append(tuple(h for h in handlelist if h.get_label()[0] != '_'))
                labels.append(str(identifier))
            primitives['legend'] = ax.legend(title=overlay.label, handles=handles, labels=labels)
        ax.autoscale(axis='x', tight=True)
        ax.set_ylim(0, None)

    return fig, ax, primitives


def plotratio(num, denom, ax=None, clear=True, overflow='none', error_opts=None, denom_fill_opts=None, guide_opts=None, unc='clopper-pearson', label=None):
    """
        Create a ratio plot, dividing two compatible histograms
        num: Hist object with single axis
        denom: Hist object with identical axis to num
        ax: matplotlib Axes object (if None, one is created)
        clear: clear Axes before drawing (if passed); if False, this function will skip drawing the legend
        overflow: overflow behavior of plot axis (see Hist.sum() docs)

        The draw options are passed as dicts to the relevant matplotlib function, with some exceptions in case
        it is especially common or useful.  If none of *_opts is specified, nothing will be plotted!
        Pass an empty dict (e.g. error_opts={}) for defaults.
            error_opts: to plot an errorbar
                Special options interpreted by this function and not passed to matplotlib:
                    'emarker' (default: '') marker to place at cap of errorbar

            denom_fill_opts: to plot a filled area centered at 1, representing denominator uncertainty
                Special options interpreted by this function and not passed to matplotlib:
                    (none)

            guide_opts: to plot a horizontal guide line at ratio of 1.
                Special options interpreted by this function and not passed to matplotlib:
                    (none)


        unc: Uncertainty calculation option
            'clopper-pearson': interval for efficiencies
            'poisson-ratio': interval for ratio of poisson distributions
            'num': poisson interval of numerator scaled by denominator value
                    (common for data/mc, for better or worse...)
        label: associate a label with this entry (note: y axis label set by num.label)
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
        if clear:
            ax.clear()
        fig = ax.figure
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
        else:
            raise ValueError("Unrecognized uncertainty option: %r" % unc)

        primitives = {}
        if error_opts is not None:
            opts = {'label': label, 'linestyle': 'none'}
            opts.update(error_opts)
            emarker = opts.pop('emarker', '')
            errbar = ax.errorbar(x=centers, y=rsumw, yerr=rsumw_err, **opts)
            plt.setp(errbar[1], 'marker', emarker)
            primitives['error'] = errbar
        if denom_fill_opts is not None:
            unity = np.ones_like(sumw_denom)
            denom_unc = poisson_interval(unity, sumw2_denom / sumw_denom**2)
            opts = {'step': 'post', 'facecolor': (0, 0, 0, 0.3), 'linewidth': 0}
            opts.update(denom_fill_opts)
            fill = ax.fill_between(edges, np.r_[denom_unc[0], denom_unc[0, -1]], np.r_[denom_unc[1], denom_unc[1, -1]], **opts)
            primitives['denom_fill'] = fill
        if guide_opts is not None:
            opts = {'linestyle': '--', 'color': (0, 0, 0, 0.5), 'linewidth': 1}
            opts.update(guide_opts)
            primitives['guide'] = ax.axhline(1., **opts)

    if clear:
        ax.autoscale(axis='x', tight=True)
        ax.set_ylim(0, None)

    return fig, ax, primitives


def plot2d(hist, xaxis, ax=None, clear=True, xoverflow='none', yoverflow='none', patch_opts=None, text_opts=None, density=False, binwnorm=None):
    """
        hist: Hist object with two dimensions
        xaxis: which of the two dimensions to use as x axis
        ax: matplotlib Axes object (if None, one is created)
        clear: clear Axes before drawing (if passed); if False, this function will skip drawing the colorbar
        xoverflow: overflow behavior of x axis (see Hist.sum() docs)
        yoverflow: overflow behavior of y axis (see Hist.sum() docs)

        The draw options are passed as dicts to the relevant matplotlib function, with some exceptions in case
        it is especially common or useful.  If none of *_opts is specified, nothing will be plotted!
        Pass an empty dict (e.g. patch_opts={}) for defaults
            patch_opts: options to plot a rectangular grid of patches colored according to the bin values
                See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pcolormesh.html for details
                Special options interpreted by this function and not passed to matplotlib:
                    (none)

            text_opts: options to draw text values at bin centers
                See https://matplotlib.org/api/text_api.html#matplotlib.text.Text for details
                Special options interpreted by this function and not passed to matplotlib:
                    'format': printf-style float format, default '%.2g'

        density: Convert sum weights to probability density (i.e. integrates to 1 over domain of axes) (NB: conflicts with binwnorm)
        binwnorm: Convert sum weights to bin-area-normalized, with units equal to supplied value (usually you want to specify 1.)
    """
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

        primitives = {}
        if patch_opts is not None:
            opts = {'cmap': 'viridis'}
            opts.update(patch_opts)
            pc = ax.pcolormesh(xedges, yedges, sumw.T, **opts)
            ax.add_collection(pc)
            primitives['patches'] = pc
            if clear:
                fig.colorbar(pc, ax=ax, label=hist.label)
        if text_opts is not None:
            primitives['texts'] = []
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
                    text = ax.text(xcenter, ycenter, txtformat % sumw[ix, iy], **opts)
                    primitives['texts'].append(text)

    if clear:
        ax.set_xlabel(xaxis.label)
        ax.set_ylabel(yaxis.label)
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(yedges[0], yedges[-1])

    return fig, ax, primitives


def plotgrid(h, figure=None, row=None, col=None, overlay=None, row_overflow='none', col_overflow='none', **plot_opts):
    """
        Create a grid of plots, enumerating identifiers on up to 3 axes:
            row: name of row axis
            col: name of column axis
            overlay: name of overlay axis
        The remaining axis will be the plot axis, with plot_opts passed to the plot1d() call

        Pass a figure object to redraw on existing figure
    """
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
            hcol = h.project(col, vcol)
            coltitle = str(vcol)
            if isinstance(vcol, Interval) and vcol.label is None:
                coltitle = "%s ∈ %s" % (h.axis(col).label, coltitle)
        for irow in range(nrow):
            ax = axes[irow, icol]
            hplot = hcol
            rowtitle = None
            if row:
                vrow = row_identifiers[irow]
                hplot = hcol.project(row, vrow)
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

    return fig, axes
