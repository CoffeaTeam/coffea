from fnal_column_analysis_tools.util import numpy as np
import json


class LumiData(object):
    """
        Class to hold and parse per-lumiSection integrated lumi values
        as returned by brilcalc, e.g. with a command such as:
        $ brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml \
                -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
                -u /pb --byls --output-style csv -i Cert_294927-306462_13TeV_PromptReco_Collisions17_JSON.txt > lumi2017.csv
    """
    def __init__(self, lumi_csv):
        self._lumidata = np.loadtxt(lumi_csv, delimiter=',', usecols=(0,1,6,7), converters={
            0: lambda s: s.split(b':')[0],
            1: lambda s: s.split(b':')[0], # not sure what lumi:0 means, appears to be always zero (DAQ off before beam dump?)
        })
        self._runlumi_type = [('run', 'u4'), ('lumi', 'u4')]
        self._index = self._lumidata[:,:2].astype('u4').view(self._runlumi_type)
        
    def get_lumi(self, runlumis):
        """
            Return integrated lumi
            runlumis: 2d numpy array of [[run,lumi], [run,lumi], ...] or LumiList object
        """
        if isinstance(runlumis, LumiList):
            runlumis = runlumis.array
        if runlumis.shape[1] != 2:
            raise TypeError("Invalid run-lumi index")
        runlumis = runlumis.astype('u4').view(self._runlumi_type)
        # numpy 1.15 introduces return_indices for intersect1d
        indices = np.isin(self._index, runlumis)[:,0]
        return self._lumidata[indices,2].sum()


class LumiMask(object):
    """
        Class that parses a 'golden json' into an efficient valid lumiSection lookup table
        Instantiate with the json file, and call with an array of runs and lumiSections, to
        return a boolean array, where valid lumiSections are marked True
    """
    def __init__(self, jsonfile):
        with open(jsonfile) as fin:
            goldenjson = json.load(fin)
        self._masks = {}
        for run, lumilist in goldenjson.items():
            run = int(run)
            mask = np.array(lumilist).flatten()
            mask[::2] -= 1
            self._masks[run] = mask

    def __call__(self, runs, lumis):
        mask = np.zeros(dtype='bool', shape=runs.shape)
        for run in np.unique(runs):
            if run in self._masks:
                mask |= (np.searchsorted(self._masks[run], lumis)%2==1) & (runs==run)
        return mask


class LumiList(object):
    """
        Mergeable (using +=) list of unique (run,lumiSection) values
        The member array can be passed to LumiData.get_lumi()
    """
    def __init__(self, runs=None, lumis=None):
        self.array = np.zeros(shape=(0,2))
        if runs is not None:
            self.array = np.unique(np.c_[runs, lumis], axis=0)
    
    def __iadd__(self, other):
        # TODO: re-apply unique? Or wait until end
        if isinstance(other, LumiList):
            self.array = np.r_[self.array, other.array]
        return self
    
    def clear(self):
        self.array = np.zeros(shape=(0,2))

