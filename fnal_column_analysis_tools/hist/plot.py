from __future__ import division
import numpy as np
import scipy.stats
import copy

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
    """
    scale = np.empty_like(sumw)
    scale[sumw!=0] = sumw2[sumw!=0] / sumw[sumw!=0]
    if np.sum(sumw==0) > 0:
        missing = np.where(sumw==0)
        available = np.nonzero(sumw)
        nearest = sum([np.subtract.outer(d,d0)**2 for d,d0 in zip(available, missing)]).argmin(axis=0)
        argnearest = tuple(dim[nearest] for dim in available)
        scale[missing] = scale[argnearest]
    counts = sumw / scale
    lo = scale * scipy.stats.chi2.ppf(scipy.stats.norm.cdf(-sigma), 2*counts) / 2.
    hi = scale * scipy.stats.chi2.ppf(scipy.stats.norm.cdf(sigma), 2*(counts+1)) / 2.
    interval = np.array([lo, hi])
    interval[interval==np.nan] = 0.  # chi2.ppf produces nan for counts=0
    return interval


def plot(ax, hist, stack=False, overflow=False, line_opts=None, fill_opts=None, error_opts=None):
    if hist.dense_dim() != 1:
        raise NotImplementedError("plot() supports only histograms with a single dense dimension")
    if not isinstance(ax, plt.Axes):
        raise ValueError("ax must be a matplotlib Axes object")
    if hist.sparse_dim() > 1:
        raise ValueError("plot() can only support up to one sparse dimension (to stack or overlay)")

    axis = hist.dense_axes()[0]
    ax.set_xlabel(axis.label)
    ax.set_ylabel(hist.label)
    edges = axis.edges(extended=overflow)
    # Only errorbar uses centers, and if we draw a step too, we need
    #   the step to go to the edge of the end bins, so place edges
    #   and only draw errorbars for the interior points
    centers = np.r_[edges[0], axis.centers(extended=overflow), edges[-1]]
    all_frequencies = hist.values(sumw2=True, overflow_view=slice(None, -1) if overflow else slice(1, -2))
    stack_sumw, stack_sumw2 = None, None
    out = {}
    for i,sparse_key in enumerate(all_frequencies.keys()):
        sumw, sumw2 = all_frequencies[sparse_key]
        # step expects edges to match frequencies (why?!)
        sumw = np.r_[sumw, sumw[-1]]
        sumw2 = np.r_[sumw2, sumw2[-1]]
        label = sparse_key[0] if len(sparse_key)>0 else hist.label
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
                y = np.r_[sumw[0], sumw[:-1], sumw[-2]]
                yerr = np.c_[np.zeros(2).reshape(2,1), err[:,:-1], np.zeros(2).reshape(2,1)]
                el = ax.errorbar(x=centers, y=y, yerr=yerr[0], uplims=True, **opts)
                opts['label'] = '_nolabel_'
                opts['linestyle'] = 'none'
                opts['color'] = el.get_children()[2].get_color()
                eh = ax.errorbar(x=centers, y=y, yerr=yerr[1], lolims=True, **opts)
                el[1][0].set_marker(emarker)
                eh[1][0].set_marker(emarker)
                out[label].append((el,eh))
    if stack_sumw is not None and error_opts is not None:
        err = poisson_interval(stack_sumw, stack_sumw2)
        opts = {'step': 'post'}
        opts.update(error_opts)
        eh = ax.fill_between(x=edges, y1=err[0,:], y2=err[1,:], step='post', **opts)
        out['stack_uncertainty'] = [eh]
    return out


def row(hist, axis):
    raise NotImplementedError("Row of plots")


def grid(hist, axis1, axis2):
    raise NotImplementedError("Grid of plots")
