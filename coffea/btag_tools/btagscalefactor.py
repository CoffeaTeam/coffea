from threading import Lock
import pandas
import numpy
import numexpr
import numba


class BTagScaleFactor:
    formulaLock = Lock()
    formulaCache = {}
    LOOSE, MEDIUM, TIGHT, RESHAPE = range(4)
    wpString = {'loose': LOOSE, 'medium': MEDIUM, 'tight': TIGHT}
    expectedColumns = [
        'OperatingPoint', 'measurementType', 'sysType', 'jetFlavor', 'etaMin',
        'etaMax', 'ptMin', 'ptMax', 'discrMin', 'discrMax', 'formula'
    ]

    @classmethod
    def readcsv(cls, filename):
        df = pandas.read_csv(filename, skipinitialspace=True)
        discriminator = df.columns[0].split(';')[0]

        def cleanup(colname):
            if ';' in colname:
                _, colname = colname.split(';')
            return colname.strip()
        df.rename(columns=cleanup, inplace=True)
        if not list(df.columns) == BTagScaleFactor.expectedColumns:
            raise RuntimeError('Columns in BTag scale factor file %s as expected' % filename)
        for var in ['eta', 'pt', 'discr']:
            df[var + 'Bin'] = list(zip(df[var + 'Min'], df[var + 'Max']))
            del df[var + 'Min']
            del df[var + 'Max']
        return df, discriminator

    def __init__(self, filename, workingpoint, lightmethod='comb', keep_df=False):
        try:
            workingpoint = BTagScaleFactor.wpString[workingpoint]
        except KeyError:
            pass
        if workingpoint not in [0, 1, 2, 3]:
            raise ValueError('Unrecognized working point')
        if lightmethod not in ['comb', 'mujets']:
            raise ValueError('Unrecognized light jet correction method')
        self.workingpoint = workingpoint
        df, self.discriminator = BTagScaleFactor.readcsv(filename)
        df = df[df['OperatingPoint'] == workingpoint]
        df = df[(df['measurementType'] == lightmethod) | (df['jetFlavor'] == 2)]
        df = df.set_index(['sysType', 'jetFlavor', 'etaBin', 'ptBin', 'discrBin']).sort_index()
        if keep_df:
            self.df = df
        self._corrections = {}
        for syst in list(df.index.levels[0]):
            corr = df.loc[syst]
            allbins = list(corr.index)
            edges_flavor = numpy.array([0, 1, 2, 3])  # udsg, c, b
            edges_eta = numpy.array(sorted(set(x for tup in corr.index.levels[1] for x in tup)))
            edges_pt = numpy.array(sorted(set(x for tup in corr.index.levels[2] for x in tup)))
            edges_discr = numpy.array(sorted(set(x for tup in corr.index.levels[3] for x in tup)))
            alledges = numpy.meshgrid(edges_flavor[:-1], edges_eta[:-1], edges_pt[:-1], edges_discr[:-1], indexing='ij')
            mapping = numpy.full(alledges[0].shape, -1)

            def findbin(flavor, eta, pt, discr):
                for i, (fbin, ebin, pbin, dbin) in enumerate(allbins):
                    if flavor == fbin and ebin[0] <= eta < ebin[1] and pbin[0] <= pt < pbin[1] and dbin[0] <= discr < dbin[1]:
                        return i
                return -1

            for idx, _ in numpy.ndenumerate(mapping):
                flavor, eta, pt, discr = (x[idx] for x in alledges)
                mapidx = findbin(flavor, eta, pt, discr)
                if mapidx < 0 and flavor == 2:
                    # it seems for b flavor it is abs(eta)
                    eta = eta + 1e-5  # add eps to avoid edge effects
                    mapidx = findbin(flavor, abs(eta), pt, discr)
                mapping[idx] = mapidx

            self._corrections[syst] = (
                edges_flavor,
                edges_eta,
                edges_pt,
                edges_discr,
                mapping,
                numpy.array(corr['formula']),
            )
        self._compiled = {}

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop('_compiled')
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self._compiled = {}

    @classmethod
    def compile(cls, formula):
        with BTagScaleFactor.formulaLock:
            try:
                return BTagScaleFactor.formulaCache[formula]
            except KeyError:
                if 'x' in formula:
                    feval = eval('lambda x: ' + formula, {'log': numpy.log, 'sqrt': numpy.sqrt})
                    out = numba.vectorize([
                        numba.float32(numba.float32),
                        numba.float64(numba.float64),
                    ])(feval)
                else:
                    val = numexpr.evaluate(formula)

                    def duck(_, out, where):
                        out[where] = val
                    out = duck
                BTagScaleFactor.formulaCache[formula] = out
                return out

    def lookup(self, axis, values):
        return numpy.clip(numpy.searchsorted(axis, values, side='right') - 1, 0, len(axis) - 2)

    def eval(self, systematic, flavor, eta, pt, discr=None, ignore_missing=False):
        try:
            functions = self._compiled[systematic]
        except KeyError:
            functions = [BTagScaleFactor.compile(f) for f in self._corrections[systematic][5]]
            self._compiled[systematic] = functions
        try:
            flavor.counts
            jin, flavor = flavor, flavor.flatten()
            eta = eta.flatten()
            pt = pt.flatten()
            discr = discr.flatten() if discr is not None else None
        except AttributeError:
            jin = None
        corr = self._corrections[systematic]
        idx = (
            self.lookup(corr[0], flavor),
            self.lookup(corr[1], eta),
            self.lookup(corr[2], pt),
            self.lookup(corr[3], discr) if discr is not None else 0,
        )
        mapidx = corr[4][idx]
        out = numpy.zeros(mapidx.shape, dtype=pt.dtype)
        for ifunc in numpy.unique(mapidx):
            if ifunc < 0 and not ignore_missing:
                raise ValueError('No correction was available for some items')
            func = functions[ifunc]
            var = discr if self.workingpoint == BTagScaleFactor.RESHAPE else pt
            func(var, out=out, where=(mapidx == ifunc))

        if jin is not None:
            out = jin.copy(content=out)
        return out
