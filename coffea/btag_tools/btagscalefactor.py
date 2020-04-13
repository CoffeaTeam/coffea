from threading import Lock
import pandas
import numpy
import numba


class BTagScaleFactor:
    '''A class holding one complete BTag scale factor for a given working point

    Parameters
    ----------
        filename : str
            The BTag-formatted CSV file to read (accepts .csv, .csv.gz, etc.)
            See pandas read_csv for all supported compressions.
        workingpoint : str or int
            The working point, one of LOOSE, MEDIUM, TIGHT, or RESHAPE (0-3, respectively)
        methods : str, optional
            The scale factor derivation method to use for each flavor, 'b,c,light' respectively.
            Defaults to 'comb,comb,incl'
        keep_df : bool, optional
            If set true, keep the parsed dataframe as an attribute (.df) for later inspection
    '''
    LOOSE, MEDIUM, TIGHT, RESHAPE = range(4)
    FLAV_B, FLAV_C, FLAV_UDSG = range(3)
    _formulaLock = Lock()
    _formulaCache = {}
    _btvflavor = numpy.array([0, 1, 2, 3])
    _flavor = numpy.array([0, 4, 5, 6])
    _wpString = {'loose': LOOSE, 'medium': MEDIUM, 'tight': TIGHT, 'reshape': RESHAPE}
    _expectedColumns = [
        'OperatingPoint', 'measurementType', 'sysType', 'jetFlavor', 'etaMin',
        'etaMax', 'ptMin', 'ptMax', 'discrMin', 'discrMax', 'formula'
    ]

    @classmethod
    def readcsv(cls, filename):
        '''Reads a BTag-formmated CSV file into a pandas dataframe

        This function also merges the bin min and max into a tuple representing the bin

        Parameters
        ----------
            filename : str
                The file to open

        Returns
        -------
            df : pandas.DataFrame
                A dataframe containing all info in the file
            discriminator : str
                The name of the discriminator the correction map is for
        '''
        df = pandas.read_csv(filename, skipinitialspace=True)
        discriminator = df.columns[0].split(';')[0]

        def cleanup(colname):
            if ';' in colname:
                _, colname = colname.split(';')
            return colname.strip()
        df.rename(columns=cleanup, inplace=True)
        if not list(df.columns) == BTagScaleFactor._expectedColumns:
            raise RuntimeError('Columns in BTag scale factor file %s as expected' % filename)
        for var in ['eta', 'pt', 'discr']:
            df[var + 'Bin'] = list(zip(df[var + 'Min'], df[var + 'Max']))
            del df[var + 'Min']
            del df[var + 'Max']
        return df, discriminator

    def __init__(self, filename, workingpoint, methods='comb,comb,incl', keep_df=False):
        if workingpoint not in [0, 1, 2, 3]:
            try:
                workingpoint = BTagScaleFactor._wpString[workingpoint.lower()]
            except (KeyError, AttributeError):
                raise ValueError('Unrecognized working point')
        methods = methods.split(',')
        self.workingpoint = workingpoint
        df, self.discriminator = BTagScaleFactor.readcsv(filename)
        cut = (df['jetFlavor'] == self.FLAV_B) & (df['measurementType'] == methods[0])
        if len(methods) > 1:
            cut |= (df['jetFlavor'] == self.FLAV_C) & (df['measurementType'] == methods[1])
        if len(methods) > 2:
            cut |= (df['jetFlavor'] == self.FLAV_UDSG) & (df['measurementType'] == methods[2])
        cut &= df['OperatingPoint'] == workingpoint
        df = df[cut]
        mavailable = list(df['measurementType'].unique())
        if not all(m in mavailable for m in methods):
            raise ValueError('Unrecognized jet correction method, available: %r' % mavailable)
        df = df.set_index(['sysType', 'jetFlavor', 'etaBin', 'ptBin', 'discrBin']).sort_index()
        if keep_df:
            self.df = df
        self._corrections = {}
        for syst in list(df.index.levels[0]):
            corr = df.loc[syst]
            allbins = list(corr.index)
            # NOTE: here we force the class to assume abs(eta) based on lack of examples where signed eta is used
            edges_eta = numpy.array(sorted(set(abs(x) for tup in corr.index.levels[1] for x in tup)))
            edges_pt = numpy.array(sorted(set(x for tup in corr.index.levels[2] for x in tup)))
            edges_discr = numpy.array(sorted(set(x for tup in corr.index.levels[3] for x in tup)))
            alledges = numpy.meshgrid(self._btvflavor[:-1], edges_eta[:-1], edges_pt[:-1], edges_discr[:-1], indexing='ij')
            mapping = numpy.full(alledges[0].shape, -1)

            def findbin(btvflavor, eta, pt, discr):
                for i, (fbin, ebin, pbin, dbin) in enumerate(allbins):
                    if btvflavor == fbin and ebin[0] <= eta < ebin[1] and pbin[0] <= pt < pbin[1] and dbin[0] <= discr < dbin[1]:
                        return i
                return -1

            for idx, _ in numpy.ndenumerate(mapping):
                btvflavor, eta, pt, discr = (x[idx] for x in alledges)
                mapping[idx] = findbin(btvflavor, eta, pt, discr)

            self._corrections[syst] = (
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
    def _compile(cls, formula):
        with BTagScaleFactor._formulaLock:
            try:
                return BTagScaleFactor._formulaCache[formula]
            except KeyError:
                if 'x' in formula:
                    feval = eval('lambda x: ' + formula, {'log': numpy.log, 'sqrt': numpy.sqrt})
                    out = numba.jit()(feval)
                else:
                    out = eval(formula)
                BTagScaleFactor._formulaCache[formula] = out
                return out

    def _lookup(self, axis, values):
        if len(axis) == 2:
            return numpy.zeros(shape=values.shape, dtype=numpy.uint)
        return numpy.clip(numpy.searchsorted(axis, values, side='right') - 1, 0, len(axis) - 2)

    def eval(self, systematic, flavor, abseta, pt, discr=None, ignore_missing=False):
        '''Evaluate this scale factor as a function of jet properties

        Parameters
        ----------
            systematic : str
                Which systematic to evaluate. Nominal correction is 'central', the options depend
                on the scale factor and method
            flavor : numpy.ndarray or awkward.Array
                The generated jet hadron flavor, following the enumeration:
                0: uds quark or gluon, 4: charm quark, 5: bottom quark
            abseta : numpy.ndarray or awkward.Array
                The absolute value of the jet pseudorapitiy
            pt : numpy.ndarray or awkward.Array
                The jet transverse momentum
            discr : numpy.ndarray or awkward.Array, optional
                The jet tagging discriminant value (default None), optional for all scale factors except
                the reshaping scale factor
            ignore_missing : bool, optional
                If set true, any values that have no correction will return 1. instead of throwing
                an exception. Out-of-bounds values are always clipped to the nearest bin.

        Returns
        -------
            out : numpy.ndarray or awkward.Array
                An array with shape matching ``pt``, containing the per-jet scale factor
        '''
        if self.workingpoint == BTagScaleFactor.RESHAPE and discr is None:
            raise ValueError('RESHAPE scale factor requires a discriminant array')
        try:
            functions = self._compiled[systematic]
        except KeyError:
            functions = [BTagScaleFactor._compile(f) for f in self._corrections[systematic][-1]]
            self._compiled[systematic] = functions

        try:
            flavor.counts
            jin, flavor = flavor, flavor.flatten()
            abseta = abseta.flatten()
            pt = pt.flatten()
            discr = discr.flatten() if discr is not None else None
        except AttributeError:
            jin = None
        corr = self._corrections[systematic]
        idx = (
            2 - self._lookup(self._flavor, flavor),  # transform to btv definiton
            self._lookup(corr[0], abseta),
            self._lookup(corr[1], pt),
            self._lookup(corr[2], discr) if discr is not None else 0,
        )
        mapidx = corr[3][idx]
        out = numpy.ones(mapidx.shape, dtype=pt.dtype)
        for ifunc in numpy.unique(mapidx):
            if ifunc < 0 and not ignore_missing:
                raise ValueError('No correction was available for some items')
            func = functions[ifunc]
            if self.workingpoint == BTagScaleFactor.RESHAPE:
                var = numpy.clip(discr, corr[2][0], corr[2][-1])
            else:
                var = numpy.clip(pt, corr[1][0], corr[1][-1])
            where = (mapidx == ifunc)
            if isinstance(func, float):
                out[where] = func
            else:
                tmp = func(var[where])
                out[where] = tmp

        if jin is not None:
            out = jin.copy(content=out)
        return out

    def __call__(self, systematic, flavor, abseta, pt, discr=None, ignore_missing=False):
        return self.eval(systematic, flavor, abseta, pt, discr, ignore_missing)
