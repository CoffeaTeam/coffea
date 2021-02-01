from ..util import numpy as np
import json
import awkward as ak

from ..util import numba
from numba import types
from numba.typed import Dict


class LumiData(object):
    r"""Holds per-lumiSection integrated lumi values

    Parameters
    ----------
        lumi_csv : str
            The path the the luminosity csv output file

    The values are extracted from the csv output as returned by brilcalc, e.g. with a command such as::

        brilcalc lumi -c /cvmfs/cms.cern.ch/SITECONF/local/JobConfig/site-local-config.xml \
                 -b "STABLE BEAMS" --normtag=/cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json \
                 -u /pb --byls --output-style csv -i Cert_294927-306462_13TeV_PromptReco_Collisions17_JSON.txt > lumi2017.csv
    """
    def __init__(self, lumi_csv):
        self._lumidata = np.loadtxt(lumi_csv, delimiter=',', usecols=(0, 1, 6, 7), converters={
            0: lambda s: s.split(b':')[0],
            1: lambda s: s.split(b':')[0],  # not sure what lumi:0 means, appears to be always zero (DAQ off before beam dump?)
        })
        self.index = Dict.empty(
            key_type=types.Tuple([types.uint32, types.uint32]),
            value_type=types.float64
        )
        self.build_lumi_table()

    def build_lumi_table(self):
        """Build index for numba-compiled functions

        This needs to be executed upon unpickling, it should be part of
        a custom deserialize function.
        """
        runs = self._lumidata[:, 0].astype('u4')
        lumis = self._lumidata[:, 1].astype('u4')
        LumiData._build_lumi_table_kernel(runs, lumis, self._lumidata, self.index)

    @staticmethod
    @numba.njit(parallel=False, fastmath=False)
    def _build_lumi_table_kernel(runs, lumis, lumidata, index):
        for i in range(len(runs)):
            run = runs[i]
            lumi = lumis[i]
            index[(run, lumi)] = float(lumidata[i, 2])

    def get_lumi(self, runlumis):
        """Calculate integrated lumi

        Parameters
        ----------
            runlumis : numpy.ndarray or LumiList
                A 2d numpy array of ``[[run,lumi], [run,lumi], ...]`` or `LumiList` object
                of the lumiSections to integrate over.
        """
        if isinstance(runlumis, LumiList):
            runlumis = runlumis.array
        tot_lumi = np.zeros((1, ), dtype=np.float64)
        LumiData._get_lumi_kernel(runlumis[:, 0], runlumis[:, 1], self.index, tot_lumi)
        return tot_lumi[0]

    @staticmethod
    @numba.njit(parallel=False, fastmath=False)
    def _get_lumi_kernel(runs, lumis, index, tot_lumi):
        ks_done = set()
        for iev in range(len(runs)):
            run = np.uint32(runs[iev])
            lumi = np.uint32(lumis[iev])
            k = (run, lumi)
            if k not in ks_done:
                ks_done.add(k)
                tot_lumi[0] += index.get(k, 0)


class LumiMask(object):
    """Holds a luminosity mask index, and provides vectorized lookup

    Parameters
    ----------
        jsonfile : str
            Path the the 'golden json' file or other valid lumiSection database in json format.

    This class parses a CMS lumi json into an efficient valid lumiSection lookup table
    """
    def __init__(self, jsonfile):
        with open(jsonfile) as fin:
            goldenjson = json.load(fin)

        self._masks = Dict.empty(
            key_type=types.uint32,
            value_type=types.uint32[:]
        )

        for run, lumilist in goldenjson.items():
            mask = np.array(lumilist, dtype=np.uint32).flatten()
            mask[::2] -= 1
            self._masks[np.uint32(run)] = mask

    def __call__(self, runs, lumis):
        """Check if run and lumi are valid

        Parameters
        ----------
            runs : numpy.ndarray
                Vectorized list of run numbers
            lumis : numpy.ndarray
                Vectorized list of lumiSection numbers

        Returns
        -------
            mask_out : numpy.ndarray
                An array of dtype `bool` where valid (run, lumi) tuples
                will have their corresponding entry set ``True``.
        """
        if isinstance(runs, ak.highlevel.Array):
            runs = ak.to_numpy(runs)
        if isinstance(lumis, ak.highlevel.Array):
            lumis = ak.to_numpy(lumis)
        mask_out = np.zeros(dtype='bool', shape=runs.shape)
        LumiMask._apply_run_lumi_mask(self._masks, runs, lumis, mask_out)
        return mask_out

    @staticmethod
    def _apply_run_lumi_mask(masks, runs, lumis, mask_out):
        LumiMask._apply_run_lumi_mask_kernel(masks, runs, lumis, mask_out)

    # This could be run in parallel, but windows does not support it
    @staticmethod
    @numba.njit(parallel=False, fastmath=True)
    def _apply_run_lumi_mask_kernel(masks, runs, lumis, mask_out):
        for iev in numba.prange(len(runs)):
            run = np.uint32(runs[iev])
            lumi = np.uint32(lumis[iev])

            if run in masks:
                lumimask = masks[run]
                ind = np.searchsorted(lumimask, lumi)
                if np.mod(ind, 2) == 1:
                    mask_out[iev] = 1


class LumiList(object):
    """Mergeable list of unique (run, lumiSection) values

    This list can be merged with another via ``+=``.

    Parameters
    ----------
        runs : numpy.ndarray
            Vectorized list of run numbers
        lumis : numpy.ndarray
            Vectorized list of lumiSection values
    """
    def __init__(self, runs=None, lumis=None):
        self.array = np.zeros(shape=(0, 2))
        if runs is not None:
            self.array = np.unique(np.c_[runs, lumis], axis=0)

    def __iadd__(self, other):
        # TODO: re-apply unique? Or wait until end
        if isinstance(other, LumiList):
            self.array = np.r_[self.array, other.array]
        else:
            raise ValueError("Expected LumiList object, got %r" % other)
        return self

    def clear(self):
        """Clear current lumi list"""
        self.array = np.zeros(shape=(0, 2))
