# -*- coding: utf-8 -*-
from __future__ import print_function, division

from coffea import hist
from coffea.util import numpy as np
import requests
import os

url = 'http://scikit-hep.org/uproot/examples/HZZ.root'
r = requests.get(url)
with open(os.path.join(os.getcwd(), 'HZZ.root'), 'wb') as f:
    f.write(r.content)

def test_import():
    from coffea.hist import plot

def fill_lepton_kinematics():
    import uproot
    import uproot_methods
    import awkward

    # histogram creation and manipulation
    from coffea import hist

    fin = uproot.open("HZZ.root")
    tree = fin["events"]

    arrays = {k.replace('Electron_', ''): v for k,v in tree.arrays("Electron_*", namedecode='ascii').items()}
    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(
                                                       arrays.pop('Px'),
                                                       arrays.pop('Py'),
                                                       arrays.pop('Pz'),
                                                       arrays.pop('E'),
                                                       )
    electrons = awkward.JaggedArray.zip(p4=p4, **arrays)

    arrays = {k.replace('Muon_', ''): v for k,v in tree.arrays("Muon_*", namedecode='ascii').items()}
    p4 = uproot_methods.TLorentzVectorArray.from_cartesian(
                                                       arrays.pop('Px'),
                                                       arrays.pop('Py'),
                                                       arrays.pop('Pz'),
                                                       arrays.pop('E'),
                                                       )
    muons = awkward.JaggedArray.zip(p4=p4, **arrays)

    # Two types of axes exist presently: bins and categories
    lepton_kinematics = hist.Hist("Events",
                                  hist.Cat("flavor", "Lepton flavor"),
                                  hist.Bin("pt", "$p_{T}$", 19, 10, 100),
                                  hist.Bin("eta", "$\eta$", [-2.5, -1.4, 0, 1.4, 2.5]),
                                  )

    # Pass keyword arguments to fill, all arrays must be flat numpy arrays
    # User is responsible for ensuring all arrays have same jagged structure!
    lepton_kinematics.fill(flavor="electron", pt=electrons['p4'].pt.flatten(), eta=electrons['p4'].eta.flatten())
    lepton_kinematics.fill(flavor="muon", pt=muons['p4'].pt.flatten(), eta=muons['p4'].eta.flatten())

    return lepton_kinematics

def test_plot1d():
    # histogram creation and manipulation
    from coffea import hist
    # matplotlib
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    
    lepton_kinematics = fill_lepton_kinematics()
    
    # looking at lepton pt for all eta
    lepton_pt = lepton_kinematics.project("eta", overflow='under')

    fig, ax, primitives = hist.plot1d(lepton_pt, overlay="flavor", stack=True,
                                      fill_opts={'alpha': .5, 'edgecolor': (0,0,0,0.3)})
    # all matplotlib primitives are returned, in case one wants to tweak them
    # e.g. maybe you really miss '90s graphics...
    primitives['legend'].shadow = True

    # Clearly the yields are much different, are the shapes similar?
    lepton_pt.label = "Density"
    fig, ax, primitives = hist.plot1d(lepton_pt, overlay="flavor", density=True)
    # ...somewhat, maybe electrons are a bit softer


def test_plot2d():
    # histogram creation and manipulation
    from coffea import hist
    # matplotlib
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    
    lepton_kinematics = fill_lepton_kinematics()
    
    # looking at lepton pt for all eta
    muon_kinematics = lepton_kinematics.project("flavor","muon")
    
    fig, ax, primitives = hist.plot2d(muon_kinematics, "eta")


def test_plotratio():
    # histogram creation and manipulation
    from coffea import hist
    # matplotlib
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    
    lepton_kinematics = fill_lepton_kinematics()

    # Add some pseudodata to a pt histogram so we can make a nice data/mc plot
    pthist = lepton_kinematics.sum('eta')
    bin_values = pthist.axis('pt').centers()
    poisson_means = pthist.sum('flavor').values()[()]
    values = np.repeat(bin_values, np.random.poisson(poisson_means))
    pthist.fill(flavor='pseudodata', pt=values)

    # Set nicer labels, by accessing the string bins' label property
    pthist.axis('flavor').index('electron').label = 'e Flavor'
    pthist.axis('flavor').index('muon').label = r'$\mu$ Flavor'
    pthist.axis('flavor').index('pseudodata').label = r'Pseudodata from e/$\mu$'

    # using regular expressions on flavor name to select just the data
    # another method would be to fill a separate data histogram
    import re
    notdata = re.compile('(?!pseudodata)')
    
    # make a nice ratio plot
    plt.rcParams.update({
                        'font.size': 14,
                        'axes.titlesize': 18,
                        'axes.labelsize': 18,
                        'xtick.labelsize': 12,
                        'ytick.labelsize': 12
                        })
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(7,7), gridspec_kw={"height_ratios": (3, 1)}, sharex=True)
    fig.subplots_adjust(hspace=.07)

    # Here is an example of setting up a color cycler to color the various fill patches
    # http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=6
    from cycler import cycler
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c']
    ax.set_prop_cycle(cycler(color=colors))

    fill_opts = {
        'edgecolor': (0,0,0,0.3),
        'alpha': 0.8
    }
    error_opts = {
        'label':'Stat. Unc.',
        'hatch':'///',
        'facecolor':'none',
        'edgecolor':(0,0,0,.5),
        'linewidth': 0
    }
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
        'emarker': '_'
    }

    hist.plot1d(pthist[notdata],
                overlay="flavor",
                ax=ax,
                clear=False,
                stack=True,
                line_opts=None,
                fill_opts=fill_opts,
                error_opts=error_opts
                )
    hist.plot1d(pthist['pseudodata'],
                overlay="flavor",
                ax=ax,
                clear=False,
                error_opts=data_err_opts
                )

    ax.autoscale(axis='x', tight=True)
    ax.set_ylim(0, None)
    ax.set_xlabel(None)
    leg = ax.legend()

    hist.plotratio(pthist['pseudodata'].sum("flavor"), pthist[notdata].sum("flavor"),
                   ax=rax,
                   error_opts=data_err_opts,
                   denom_fill_opts={},
                   guide_opts={},
                   unc='num'
                   )
    rax.set_ylabel('Ratio')
    rax.set_ylim(0,2)

    coffee = plt.text(0., 1., u"☕",
                      fontsize=28,
                      horizontalalignment='left',
                      verticalalignment='bottom',
                      transform=ax.transAxes
                      )
    lumi = plt.text(1., 1., r"1 fb$^{-1}$ (?? TeV)",
                    fontsize=16,
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=ax.transAxes
                    )


def test_plotgrid():
    # histogram creation and manipulation
    from coffea import hist
    # matplotlib
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    
    lepton_kinematics = fill_lepton_kinematics()
    
    # Let's stack them, after defining some nice styling
    stack_fill_opts = {'alpha': 0.8, 'edgecolor':(0,0,0,.5)}
    stack_error_opts = {'label':'Stat. Unc.', 'hatch':'///', 'facecolor':'none', 'edgecolor':(0,0,0,.5), 'linewidth': 0}
    # maybe we want to compare different eta regions
    # plotgrid accepts row and column axes, and creates a grid of 1d plots as appropriate
    fig, ax = hist.plotgrid(lepton_kinematics, row="eta", overlay="flavor", stack=True,
                            fill_opts=stack_fill_opts,
                            error_opts=stack_error_opts,
                            )

def test_clopper_pearson_interval():
    from coffea.hist.plot import clopper_pearson_interval

    # Reference values for CL=0.6800 calculated with ROOT's TEfficiency
    num = np.array([1., 5., 10., 10.])
    denom = np.array([10., 10., 10., 437.])
    ref_hi = np.array([0.293313782248242, 0.6944224231766912, 1.0, 0.032438865381336446])
    ref_lo = np.array([0.01728422272382846, 0.3055775768233088, 0.8325532074018731, 0.015839046981153772])

    interval = clopper_pearson_interval(num, denom, coverage=0.68)

    threshold = 1e-6
    assert(all((interval[1, :] / ref_hi) - 1 < threshold))
    assert(all((interval[0, :] / ref_lo) - 1 < threshold))
