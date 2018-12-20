import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import copy

from .hist_tools import Hist, Bin, Cat

# Plotting is always terrible
# Let's try our best to follow matplotlib idioms
# https://matplotlib.org/tutorials/introductory/usage.html#coding-styles

def poisson_interval(vec, sigma=1):
    """
        The so-called 'exact' interval
        c.f. http://ms.mcmaster.ca/peter/s743/poissonalpha.html
    """
    return np.array([scipy.stats.chi2.ppf(scipy.stats.norm.cdf(-sigma), 2*vec), scipy.stats.chi2.ppf(scipy.stats.norm.cdf(sigma), 2*(vec+1))])/2

def centers(bins):
    return bins[:-1] + np.diff(bins)/2.

def plot(ax, hist, stack=False, param_dict=None):
    if hist.dense_dim() != 1:
        raise NotImplementedError("plot() supports only histograms with a single dense dimension")

    if param_dict is None:
        param_dict = {}

    if isinstance(ax, plt.Axes) and hist.sparse_dim() == 0:
        raise NotImplementedError("Single dense dimension")
    elif isinstance(ax, plt.Axes) and hist.sparse_dim() == 1:
        bins = hist._dense_dims[0].bin_boundaries()
        all_frequencies = hist.values(sumw2=True, overflow_view=slice(1,-1))
        stacked_freq = None
        out = {}
        for i,sparse_key in enumerate(all_frequencies.keys()):
            freq = all_frequencies[sparse_key]
            this_dict = copy.copy(param_dict)
            this_dict['label'] = sparse_key[0]
            if stack:
                if stacked_freq is None:
                    stacked_freq = copy.deepcopy(freq)
                else:
                    stacked_freq = (stacked_freq[0]+freq[0], stacked_freq[1]+freq[1])
                freq = stacked_freq
            sumw, sumw2 = freq
            l = ax.step(x=bins, y=sumw, where='post', **this_dict)
            # TODO: fill
            #f = ax.fill_between(x=bins, y1=sumw, step='post', alpha=0.4, label='_nolegend_')
            if (stack and i==len(all_frequencies)-1) or (not stack):
                gmN_scale = np.where(sumw>0,sumw2/np.maximum(sumw,1),1)
                err = poisson_interval(sumw / gmN_scale) * gmN_scale
                err = np.abs(err-sumw)
                cap = ''
                this_dict['linestyle'] = 'None'
                this_dict['color'] = l[0].get_color()
                #this_dict['capsize'] = 0.
                this_dict['label'] = '_nolegend_'
                el = ax.errorbar(x=centers(bins), y=sumw[:-1], yerr=err[0,:-1], uplims=True, **this_dict)
                el[1][0].set_marker(cap)
                eh = ax.errorbar(x=centers(bins), y=sumw[:-1], yerr=err[1,:-1], lolims=True, **this_dict)
                eh[1][0].set_marker(cap)
                out[sparse_key[0]] = (l,el,eh)
        return out
    elif isinstance(ax, list) and isinstance(ax[0], plt.Axes) and hist.sparse_dim() == 2:
        raise NotImplementedError("List of plots")
    elif isinstance(ax, list) and isinstance(ax[0], list) and hist.sparse_dim() == 3:
        raise NotImplementedError("Grid of plots")
    else:
        raise ValueError("Cannot find a way to plot this histogram")
