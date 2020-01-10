import os
import numpy as np
# crystalball is single sided, local reimplementation of double-sided here until
# the PR can be merged
#from scipy.stats import crystalball
from .doublecrystalball import doublecrystalball
from awkward import JaggedArray
from coffea.lookup_tools.dense_lookup import dense_lookup


class rochester_lookup():

    def __init__(self, path, loaduncs=False):
        self._path = path
        self._loaduncs = loaduncs
        self._read()

    def _read(self):
        initialized = False

        with open(self._path) as f:
            for line in f:
                line = line.strip('\n')
                # the number of sets available
                if line.startswith('NSET'):
                    self.nsets = int(line.split()[1])
                # the number of members in a given set
                elif line.startswith('NMEM'):
                    self.members = [int(x) for x in line.split()[1:]]
                    assert(len(self.members) == self.nsets)
                # the type of the values provided: 0 is default, 1 is replicas (for stat unc), 2 is Symhes (for various systematics)
                elif line.startswith('TVAR'):
                    self.tvars = [int(x) for x in line.split()[1:]]
                    assert(len(self.tvars) == self.nsets)
                # number of phi bins
                elif line.startswith('CPHI'):
                    self.nphi = int(line.split()[1])
                    self.phiedges = [float(x) * 2 * np.pi / self.nphi - np.pi for x in range(self.nphi + 1)]
                # number of eta bins and edges
                elif line.startswith('CETA'):
                    self.neta = int(line.split()[1])
                    self.etaedges = [float(x) for x in line.split()[2:]]
                    assert(len(self.etaedges) == self.neta + 1)
                # minimum number of tracker layers with measurement
                elif line.startswith('RMIN'):
                    self.nmin = int(line.split()[1])
                # number of bins in the number of tracker layers measurements
                elif line.startswith('RTRK'):
                    self.ntrk = int(line.split()[1])
                # number of abseta bins and edges
                elif line.startswith('RETA'):
                    self.nabseta = int(line.split()[1])
                    self.absetaedges = [float(x) for x in line.split()[2:]]
                    assert(len(self.absetaedges) == self.nabseta + 1)
                # load the parameters
                # the structure will be
                # SETNUMBER MEMBERNUMBER TAG[T,R,F,C] [TAG specific indices] [values]
                else:
                    if not initialized:
                        # don't want to necessarily load uncertainties
                        toload = self.nsets if self._loaduncs else 1
                        M = {s: {m: {t: {} for t in range(2)} for m in range(self.members[s])} for s in range(toload)}
                        A = {s: {m: {t: {} for t in range(2)} for m in range(self.members[s])} for s in range(toload)}
                        kRes = {s: {m: {t: [] for t in range(2)} for m in range(self.members[s])} for s in range(toload)}
                        rsPars = {s: {m: {t: {} for t in range(3)} for m in range(self.members[s])} for s in range(toload)}
                        cbS = {s: {m: {} for m in range(self.members[s])} for s in range(toload)}
                        cbA = {s: {m: {} for m in range(self.members[s])} for s in range(toload)}
                        cbN = {s: {m: {} for m in range(self.members[s])} for s in range(toload)}
                        initialized = True
                    remainder = line.split()
                    setn, membern, tag, *remainder = remainder
                    setn = int(setn)
                    membern = int(membern)
                    tag = str(tag)
                    # tag T has 2 indices corresponding to TYPE BINNUMBER and has RTRK+1 values each
                    # these correspond to the nTrk[2] parameters of RocRes (and BINNUMBER is the abseta bin)
                    if tag == 'T':
                        t, b, *remainder = remainder
                        t = int(t)
                        b = int(b)
                        values = [float(x) for x in remainder]
                        assert(len(values) == self.ntrk + 1)

                    # tag R has 2 indices corresponding to VARIABLE BINNUMBER and has RTRK values each
                    # these variables correspond to the rsPar[3] and crystal ball (std::vector<CrystalBall> cb) of RocRes where CrystalBall has valus s, a, n
                    # (and BINNUMBER is the abseta bin)
                    # Note: crystal ball here is a symmetric double-sided crystal ball
                    elif tag == 'R':
                        v, b, *remainder = remainder
                        v = int(v)
                        b = int(b)
                        values = [float(x) for x in remainder]
                        assert(len(values) == self.ntrk)
                        if v in range(3):
                            if setn in rsPars:
                                rsPars[setn][membern][v][b] = values
                                if v == 2:
                                    rsPars[setn][membern][v][b] = [x / 100 for x in values]
                        elif v == 3:
                            if setn in cbS:
                                cbS[setn][membern][b] = values
                        elif v == 4:
                            if setn in cbA:
                                cbA[setn][membern][b] = values
                        elif v == 5:
                            if setn in cbN:
                                cbN[setn][membern][b] = values

                    # tag F has 1 index corresponding to TYPE and has RETA values each
                    # these correspond to the kRes[2] of RocRes
                    elif tag == 'F':
                        t, *remainder = remainder
                        t = int(t)
                        values = [float(x) for x in remainder]
                        assert(len(values) == self.nabseta)
                        if setn in kRes:
                            kRes[setn][membern][t] = values

                    # tag C has 3 indices corresponding to TYPE VARIABLE BINNUMBER and has NPHI values each
                    # these correspond to M and A values of CorParams (and BINNUMBER is the eta bin)
                    # These are what are used to get the scale factor for kScaleDT (and kScaleMC)
                    #       scale = 1.0 / (M+Q*A*pT)
                    # For the kSpreadMC (gen matched, recommended) and kSmearMC (not gen matched), we need all of the above parameters
                    elif tag == 'C':
                        t, v, b, *remainder = remainder
                        t = int(t)
                        v = int(v)
                        b = int(b)
                        values = [float(x) for x in remainder]
                        assert(len(values) == self.nphi)
                        if v == 0:
                            if setn in M:
                                M[setn][membern][t][b] = [1.0 + x / 100 for x in values]
                        elif v == 1:
                            if setn in A:
                                A[setn][membern][t][b] = [x / 100 for x in values]

                    else:
                        raise ValueError(line)

        # now build the lookup tables
        # for data scale, simple, just M A in bins of eta,phi
        edges = (np.array(self.etaedges), np.array(self.phiedges))
        self._M = {s: {m: {t: dense_lookup(np.array([M[s][m][t][b] for b in range(self.neta)]), edges) for t in M[s][m]} for m in M[s]} for s in M}
        self._A = {s: {m: {t: dense_lookup(np.array([A[s][m][t][b] for b in range(self.neta)]), edges) for t in A[s][m]} for m in A[s]} for s in A}

        # for mc scale, more complicated
        # version 1 if gen pt available
        # only requires the kRes lookup
        edges = np.array(self.absetaedges)
        self._kRes = {s: {m: {t: dense_lookup(np.array(kRes[s][m][t]), edges) for t in kRes[s][m]} for m in kRes[s]} for s in kRes}

        # version 2 if gen pt not available
        self.trkedges = [0] + [self.nmin + x + 0.5 for x in range(self.ntrk)]
        edges = (np.array(self.absetaedges), np.array(self.trkedges))
        self._rsPars = {s: {m: {t: dense_lookup(np.array([rsPars[s][m][t][b] for b in range(self.nabseta)]), edges) for t in rsPars[s][m]}
                        for m in rsPars[s]} for s in rsPars}
        self._cbS = {s: {m: dense_lookup(np.array([cbS[s][m][b] for b in range(self.nabseta)]), edges) for m in cbS[s]} for s in cbS}
        self._cbA = {s: {m: dense_lookup(np.array([cbA[s][m][b] for b in range(self.nabseta)]), edges) for m in cbA[s]} for s in cbA}
        self._cbN = {s: {m: dense_lookup(np.array([cbN[s][m][b] for b in range(self.nabseta)]), edges) for m in cbN[s]} for s in cbN}

    def _error(self, func, *args):
        if not self._loaduncs:
            return None

        newargs = args + (0, 0)
        default = func(*newargs)
        result = np.zeros_like(default)
        for s in range(self.nsets):
            for m in range(self.members[s]):
                newargs = args + (s, m)
                d = func(*newargs)
                result = result + d * d / self.members[s]
        return result**0.5

    def kScaleDT(self, charge, pt, eta, phi, s=0, m=0):
        '''Momentum scale correction for data
            required:
                charge
                pt
                eta
                phi
            optional:
                s: Rochester correction set to use (default 0)
                m: Rochester correction member to use (default 0)
        '''
        # type = 1 corresponds to data
        M = self._M[s][m][1](eta, phi)
        A = self._A[s][m][1](eta, phi)
        return 1.0 / (M + charge * A * pt)

    def kScaleDTerror(self, charge, pt, eta, phi):
        '''Momentum scale correction uncertainty for data
            required:
                charge
                pt
                eta
                phi
        '''
        return self._error(self.kScaleDT, charge, pt, eta, phi)

    def kScaleMC(self, charge, pt, eta, phi, s=0, m=0):
        '''Momentum scale correction for mc (not recommended, use kSpreadMC instead)
            required:
                charge
                pt
                eta
                phi
            optional:
                s: Rochester correction set to use (default 0)
                m: Rochester correction member to use (default 0)
        '''
        # type = 0 corresponds to mc
        M = self._M[s][m][0](eta, phi)
        A = self._A[s][m][0](eta, phi)
        return 1.0 / (M + charge * A * pt)

    def kScaleMCerror(self, charge, pt, eta, phi):
        '''Momentum scale correction uncertainty for mc (not recommended, use kSpreadMC instead)
            required:
                charge
                pt
                eta
                phi
        '''
        return self._error(self.kScaleMC, charge, pt, eta, phi)

    def kSpreadMC(self, charge, pt, eta, phi, genpt, s=0, m=0):
        '''Momentum scale correction for mc (if genpt not availble, use kSmearMC)
            required:
                charge
                pt
                eta
                phi
                genpt: the match gen particle pt
            optional:
                s: Rochester correction set to use (default 0)
                m: Rochester correction member to use (default 0)
        '''
        k = self.kScaleMC(charge, pt, eta, phi, s, m)
        return k * self._kSpread(genpt, k * pt, eta, s, m)

    def kSpreadMCerror(self, charge, pt, eta, phi, genpt):
        '''Momentum scale correction uncertainty for mc (if genpt not availble, use kSmearMC)
            required:
                charge
                pt
                eta
                phi
                genpt: the match gen particle pt
        '''
        return self._error(self.kSpreadMC, charge, pt, eta, phi, genpt)

    def _kSpread(self, genpt, kpt, eta, s=0, m=0):
        x = genpt / kpt
        kData = self._kRes[s][m][1](abs(eta))  # type 1 is data
        kMC = self._kRes[s][m][0](abs(eta))    # type 0 is MC
        return x / (1.0 + (x - 1.0) * kData / kMC)

    def kSmearMC(self, charge, pt, eta, phi, nl, u, s=0, m=0):
        '''Momentum scale correction for mc (should prefer to use kSpreadMC)
            required:
                charge
                pt
                eta
                phi
                nl: number of tracker layers with measurements
                u: random float between 0 and 1
            optional:
                s: Rochester correction set to use (default 0)
                m: Rochester correction member to use (default 0)
        '''
        k = self.kScaleMC(charge, pt, eta, phi, s, m)
        return k * self._kExtra(k * pt, eta, nl, u, s, m)

    def kSmearMCerror(self, charge, pt, eta, phi, nl, u):
        '''Momentum scale correction uncertainty for mc (should prefer to use kSpreadMC)
            required:
                charge
                pt
                eta
                phi
                nl: number of tracker layers with measurements
                u: random float between 0 and 1
        '''
        return self._error(self.kSmearMC, charge, pt, eta, phi, nl, u)

    def _sigma(self, pt, eta, nl, s=0, m=0):
        dpt = pt - 45
        return self._rsPars[s][m][0](abs(eta), nl) + self._rsPars[s][m][1](abs(eta), nl) * dpt + self._rsPars[s][m][2](abs(eta), nl) * dpt**2

    def _kExtra(self, kpt, eta, nl, u, s=0, m=0):
        # if it is a jagged array, save the offsets then flatten everything
        # needed for the ternary conditions later
        offsets = None
        if isinstance(kpt, JaggedArray):
            offsets = kpt.offsets
            kpt = kpt.flatten()
            eta = eta.flatten()
            nl = nl.flatten()
            u = u.flatten()
        kData = self._kRes[s][m][1](abs(eta))  # type 1 is data
        kMC = self._kRes[s][m][0](abs(eta))    # type 0 is MC
        mask = (kData > kMC)
        x = np.zeros_like(kpt)
        sigma = self._sigma(kpt, eta, nl, s, m)
        # Rochester cbA = beta, cbN = m, as well as cbM (always 0?) = loc and cbS = scale to transform y = (x-loc)/scale in the pdf method
        cbA = self._cbA[s][m](abs(eta), nl)
        cbN = self._cbN[s][m](abs(eta), nl)
        loc = np.zeros_like(u)
        cbS = self._cbS[s][m](abs(eta), nl)
        invcdf = doublecrystalball.ppf(u, cbA, cbA, cbN, cbN, loc, cbS)
        x[mask] = np.sqrt(kData[mask]**2 - kMC[mask]**2) * sigma[mask] * invcdf[mask]
        result = np.ones_like(kpt)
        result[(x > -1)] = 1.0 / (1.0 + x[x > -1])
        if offsets is not None:
            result = JaggedArray.fromoffsets(offsets, result)
        return result
