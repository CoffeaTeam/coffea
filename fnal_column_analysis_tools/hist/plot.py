# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import scipy.stats
import copy
import warnings
import numbers
from .hist_tools import SparseAxis, DenseAxis, overflow_behavior, Interval

import matplotlib.pyplot as plt

# Plotting is always terrible
# Let's try our best to follow matplotlib idioms
# https://matplotlib.org/tutorials/introductory/usage.html#coding-styles

def poisson_interval(sumw, sumw2, sigma=1):
    """
        The so-called 'exact' interval
            c.f. http://ms.mcmaster.ca/peter/s743/poissonalpha.html
        For weighted data, approximate the observed count by sumw**2/sumw2
            When a bin is zero, find the scale of the nearest nonzero bin
            If all bins zero, raise warning and set interval to sumw
    """
    scale = np.empty_like(sumw)
    scale[sumw!=0] = sumw2[sumw!=0] / sumw[sumw!=0]
    if np.sum(sumw==0) > 0:
        missing = np.where(sumw==0)
        available = np.nonzero(sumw)
        if len(available[0]) == 0:
            warnings.warn("All sumw are zero!  Cannot compute meaningful error bars", RuntimeWarning)
            return np.vstack([sumw, sumw])
        nearest = sum([np.subtract.outer(d,d0)**2 for d,d0 in zip(available, missing)]).argmin(axis=0)
        argnearest = tuple(dim[nearest] for dim in available)
        scale[missing] = scale[argnearest]
    counts = sumw / scale
    lo = scale * scipy.stats.chi2.ppf(scipy.stats.norm.cdf(-sigma), 2*counts) / 2.
    hi = scale * scipy.stats.chi2.ppf(scipy.stats.norm.cdf(sigma), 2*(counts+1)) / 2.
    interval = np.array([lo, hi])
    interval[interval==np.nan] = 0.  # chi2.ppf produces nan for counts=0
    return interval


def plot1d(ax, hist, axis, stack=False, overflow='none', line_opts=None, fill_opts=None, error_opts=None, overlay_overflow='none', density=False, binwnorm=None):
    """
        ax: matplotlib Axes object
        hist: Hist object with maximum of two dimensions
        axis: the axis of hist we want to use as the x axis
        stack: whether to stack or overlay the other dimension (if one exists)
        overflow: overflow behavior of plot axis (see Hist.sum() docs)

        The draw options are passed as dicts.  If none of *_opts is specified, nothing will be plotted!
        Pass an empty dict (e.g. line_opts={}) for defaults
            line_opts: options to plot a step without errors
            fill_opts: to plot a filled area
            error_opts: to plot an errorbar, with a step or marker

        overlay_overflow: overflow behavior of dense overlay axis, if one exists
        density: Convert sum weights to probability density (i.e. integrates to 1 over domain of axis) (NB: conflicts with binwnorm)
        binwnorm: Convert sum weights to bin-width-normalized, with units equal to supplied value (usually you want to specify 1.)
    """
    if not isinstance(ax, plt.Axes):
        raise ValueError("ax must be a matplotlib Axes object")
    if hist.dim() > 2:
        raise ValueError("plot1d() can only support up to two dimensions (one for axis, one to stack or overlay)")
    if density and binwnorm is not None:
        raise ValueError("Cannot use density and binwnorm at the same time!")
    if binwnorm is not None:
        if not isinstance(binwnorm, numbers.Number):
            raise ValueError("Bin width normalization not a number, but a %r" % binwnorm.__class__)

    axis = hist.axis(axis)
    other_axis = next((ax for ax in hist.axes() if ax != axis), None)
    if isinstance(axis, SparseAxis):
        raise NotImplementedError("Plot a sparse axis (e.g. bar chart)")
    elif isinstance(axis, DenseAxis):
        ax.set_xlabel(axis.label)
        ax.set_ylabel(hist.label)
        edges = axis.edges(overflow=overflow)
        # Only errorbar uses centers, and if we draw a step too, we need
        #   the step to go to the edge of the end bins, so place edges
        #   and only draw errorbars for the interior points
        centers = np.r_[edges[0], axis.centers(overflow=overflow), edges[-1]]
        # but if there's a marker, then it shows up in the extra spots
        center_view = slice(1, -1) if error_opts is not None and 'marker' in error_opts else slice(None)
        stack_sumw, stack_sumw2 = None, None
        out = {}
        identifiers = hist.identifiers(other_axis, overflow=overlay_overflow) if other_axis else [None]
        for i, identifier in enumerate(identifiers):
            if identifier is None:
                sumw, sumw2 = hist.values(sumw2=True, overflow=overflow)[()]
            elif isinstance(other_axis, SparseAxis):
                sumw, sumw2 = hist.project(other_axis, identifier).values(sumw2=True, overflow=overflow)[()]
            else:
                sumw, sumw2 = hist.values(sumw2=True, overflow='allnan')[()]
                the_slice = (i if overflow_behavior(overlay_overflow).start is None else i+1, overflow_behavior(overflow))
                if hist._idense(other_axis) == 1:
                    the_slice = (the_slice[1], the_slice[0])
                sumw = sumw[the_slice]
                sumw2 = sumw2[the_slice]
            if (density or binwnorm is not None) and np.sum(sumw)>0:
                overallnorm = np.sum(sumw)*binwnorm if binwnorm is not None else 1.
                binnorms = overallnorm / (np.diff(edges)*np.sum(sumw))
                sumw = sumw * binnorms
                sumw2 = sumw2 * binnorms**2
            # step expects edges to match frequencies (why?!)
            sumw = np.r_[sumw, sumw[-1]]
            sumw2 = np.r_[sumw2, sumw2[-1]]
            label = str(identifier)
            out[label] = []
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
                    l = ax.step(x=edges, y=stack_sumw, **opts)
                    first_color = l[0].get_color()
                    out[label].append(l)
                if fill_opts is not None:
                    opts = {'step': 'post', 'label': label}
                    if first_color is not None:
                        opts['color'] = first_color
                    opts.update(fill_opts)
                    f = ax.fill_between(x=edges, y1=stack_sumw-sumw, y2=stack_sumw, **opts)
                    if first_color is None:
                        first_color = f.get_facecolor()[0]
                    out[label].append(f)
            else:
                if line_opts is not None:
                    opts = {'where': 'post', 'label': label}
                    opts.update(line_opts)
                    l = ax.step(x=edges, y=sumw, **opts)
                    first_color = l[0].get_color()
                    out[label].append(l)
                if fill_opts is not None:
                    opts = {'step': 'post', 'label': label}
                    if first_color is not None:
                        opts['color'] = first_color
                    opts.update(fill_opts)
                    f = ax.fill_between(x=edges, y1=sumw, **opts)
                    if first_color is None:
                        first_color = f.get_facecolor()[0]
                    out[label].append(f)
                if error_opts is not None:
                    err = np.abs(poisson_interval(sumw, sumw2) - sumw)
                    emarker = error_opts.pop('emarker', '')
                    opts = {'label': label, 'drawstyle': 'steps-mid'}
                    if first_color is not None:
                        opts['color'] = first_color
                    opts.update(error_opts)
                    y = np.r_[sumw[0], sumw]
                    yerr = np.c_[np.zeros(2).reshape(2,1), err[:,:-1], np.zeros(2).reshape(2,1)]
                    el = ax.errorbar(x=centers[center_view], y=y[center_view], yerr=yerr[0,center_view], uplims=True, **opts)
                    opts['label'] = '_nolabel_'
                    opts['linestyle'] = 'none'
                    opts['color'] = el.get_children()[2].get_color()[0]
                    eh = ax.errorbar(x=centers[center_view], y=y[center_view], yerr=yerr[1,center_view], lolims=True, **opts)
                    el[1][0].set_marker(emarker)
                    eh[1][0].set_marker(emarker)
                    out[label].append((el,eh))
        if stack_sumw is not None and error_opts is not None:
            err = poisson_interval(stack_sumw, stack_sumw2)
            opts = {'step': 'post'}
            opts.update(error_opts)
            eh = ax.fill_between(x=edges, y1=err[0,:], y2=err[1,:], **opts)
            out['stack_uncertainty'] = [eh]
        return out


def plotgrid(h, row=None, col=None, overlay=None, row_overflow='none', col_overflow='none', figsize=(4,4), **plot_opts):
    """
        Create a grid of plots, enumerating identifiers on up to 3 axes:
            row: name of row axis
            col: name of column axis
            overlay: name of overlay axis
        The remaining axis will be passed to plot1d as the plot axis
    """
    haxes = set(ax.name for ax in h.axes())
    nrow,ncol = 1,1
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
    plotaxis = haxes.pop()

    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize[0]*ncol,figsize[1]*nrow), squeeze=False, sharex=True, sharey=True)

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
            ax = axes[irow][icol]
            hplot = hcol
            rowtitle = None
            if row:
                vrow = row_identifiers[irow]
                hplot = hcol.project(row, vrow)
                rowtitle = str(vrow)
                if isinstance(vrow, Interval) and vrow.label is None:
                    rowtitle = "%s ∈ %s" % (h.axis(row).label, rowtitle)

            plot1d(ax, hplot, plotaxis, **plot_opts)
            if row is not None and col is not None:
                ax.set_title("%s, %s" % (rowtitle, coltitle))
            elif row is not None:
                ax.set_title(rowtitle)
            elif col is not None:
                ax.set_title(coltitle)
            ax.autoscale(axis='x', tight=True)
            ax.set_ylim(0, None)
            if overlay is not None:
                ax.legend(title=h.axis(overlay).label)
    return fig, axes
