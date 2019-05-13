from fnal_column_analysis_tools.util import awkward
from fnal_column_analysis_tools.util import numpy as np
import os
try:
    import cStringIO as io
except ImportError:
    import io

# for later
# func = numbaize(formula,['p%i'%i for i in range(nParms)]+[varnames[i] for i in range(nEvalVars)])


def _parse_jme_formatted_file(jmeFilePath, interpolatedFunc=False, parmsFromColumns=False, jme_f=None):
    if jme_f is None:
        jme_f = open(jmeFilePath, 'r')
    layoutstr = jme_f.readline().strip().strip('{}')

    name = jmeFilePath.split('/')[-1].split('.')[0]

    layout = layoutstr.split()
    if not layout[0].isdigit():
        raise Exception('First column of JME File Header must be a digit!')

    # setup the file format
    nBinnedVars = int(layout[0])
    nBinColumns = 2 * nBinnedVars
    nEvalVars = int(layout[nBinnedVars + 1])
    formula = layout[nBinnedVars + nEvalVars + 2]
    nParms = 0
    while(formula.count('[%i]' % nParms)):
        formula = formula.replace('[%i]' % nParms, 'p%i' % nParms)
        nParms += 1
    # get rid of TMath
    tmath = {'TMath::Max': 'max', 'TMath::Log': 'log', 'TMath::Power': 'pow'}
    for key, rpl in tmath.items():
        formula = formula.replace(key, rpl)
    # protect function names with vars in them
    funcs_to_cap = ['max', 'exp', 'pow']
    for f in funcs_to_cap:
        formula = formula.replace(f, f.upper())

    templatevars = ['x', 'y', 'z', 'w', 't', 's']
    varnames = [layout[i + nBinnedVars + 2] for i in range(nEvalVars)]
    for find, replace in zip(templatevars, varnames):
        formula = formula.replace(find, replace)
    # restore max
    for f in funcs_to_cap:
        formula = formula.replace(f.upper(), f)

    # parse the columns
    minMax = ['Min', 'Max']
    columns = []
    dtypes = []
    offset = 1
    for i in range(nBinnedVars):
        columns.extend(['%s%s' % (layout[i + offset], mm) for mm in minMax])
        dtypes.extend(['<f8', '<f8'])
    columns.append('NVars')
    dtypes.append('<i8')
    offset += nBinnedVars + 1
    if not interpolatedFunc:
        for i in range(nEvalVars):
            columns.extend(['%s%s' % (layout[i + offset], mm) for mm in minMax])
            dtypes.extend(['<f8', '<f8'])
    for i in range(nParms):
        columns.append('p%i' % i)
        dtypes.append('<f8')

    if parmsFromColumns:
        pars = np.genfromtxt(jme_f, encoding='ascii')
        nParms = pars.shape[1] - len(columns)
        for i in range(nParms):
            columns.append('p%i' % i)
            dtypes.append('<f8')
        pars = np.core.records.fromarrays(
            pars.transpose(), names=columns, formats=dtypes)
    else:
        pars = np.genfromtxt(jme_f,
                             dtype=tuple(dtypes),
                             names=tuple(columns),
                             unpack=True,
                             encoding='ascii'
                         )

    outs = [name, layout, pars, nBinnedVars, nBinColumns,
            nEvalVars, formula, nParms, columns, dtypes]
    jme_f.close()
    return tuple(outs)


def _build_standard_jme_lookup(name, layout, pars, nBinnedVars, nBinColumns,
                               nEvalVars, formula, nParms, columns, dtypes,
                               interpolatedFunc=False):
    # the first bin is always usual for JECs
    # the next bins may vary in number, so they're jagged arrays... yay
    bins = {}
    offset_col = 0
    offset_name = 1
    bin_order = []
    for i in range(nBinnedVars):
        binMins = None
        binMaxs = None
        if i == 0:
            binMins = np.unique(pars[columns[0]])
            binMaxs = np.unique(pars[columns[1]])
            bins[layout[i + offset_name]] = np.union1d(binMins, binMaxs)
        else:
            counts = np.zeros(0, dtype=np.int)
            allBins = np.zeros(0, dtype=np.double)
            for binMin in bins[bin_order[0]][:-1]:
                binMins = np.unique(pars[np.where(pars[columns[0]] == binMin)][columns[i + offset_col]])
                binMaxs = np.unique(pars[np.where(pars[columns[0]] == binMin)][columns[i + offset_col + 1]])
                theBins = np.union1d(binMins, binMaxs)
                allBins = np.append(allBins, theBins)
                counts = np.append(counts, theBins.size)
            bins[layout[i + offset_name]] = awkward.JaggedArray.fromcounts(counts, allBins)
        bin_order.append(layout[i + offset_name])
        offset_col += 1

    # skip nvars to the variable columns
    # the columns here define clamps for the variables defined in columns[]
    # ----> clamps can be different from bins
    # ----> if there is more than one binning variable this array is jagged
    # ----> just make it jagged all the time
    clamp_mins = {}
    clamp_maxs = {}
    var_order = []
    offset_col = 2 * nBinnedVars + 1
    offset_name = nBinnedVars + 2
    jagged_counts = np.ones(bins[bin_order[0]].size - 1, dtype=np.int)
    if len(bin_order) > 1:
        jagged_counts = np.maximum(bins[bin_order[1]].counts - 1, 0)  # need counts-1 since we only care about Nbins
    for i in range(nEvalVars):
        var_order.append(layout[i + offset_name])
        if not interpolatedFunc:
            clamp_mins[layout[i + offset_name]] = awkward.JaggedArray.fromcounts(jagged_counts, np.atleast_1d(pars[columns[i + offset_col]]))
            clamp_maxs[layout[i + offset_name]] = awkward.JaggedArray.fromcounts(jagged_counts, np.atleast_1d(pars[columns[i + offset_col + 1]]))
            offset_col += 1

    # now get the parameters, which we will look up with the clamped values
    parms = []
    parm_order = []
    offset_col = 2 * nBinnedVars + 1 + int(not interpolatedFunc) * 2 * nEvalVars
    for i in range(nParms):
        parms.append(awkward.JaggedArray.fromcounts(jagged_counts, pars[columns[i + offset_col]]))
        parm_order.append('p%i' % (i))

    wrapped_up = {}
    wrapped_up[(name, 'jme_standard_function')] = (formula,
                                                   (bins, bin_order),
                                                   (clamp_mins, clamp_maxs, var_order),
                                                   (parms, parm_order))
    return wrapped_up


def _convert_standard_jme_txt_file(jmeFilePath):
    return _build_standard_jme_lookup(*_parse_jme_formatted_file(jmeFilePath))


convert_jec_txt_file = _convert_standard_jme_txt_file
convert_jr_txt_file = _convert_standard_jme_txt_file


def convert_jersf_txt_file(jersfFilePath):
    name, layout, pars, nBinnedVars, \
        nBinColumns, nEvalVars, formula, \
        nParms, columns, dtypes = _parse_jme_formatted_file(jersfFilePath,
                                                            parmsFromColumns=True)

    temp = _build_standard_jme_lookup(name, layout, pars, nBinnedVars, nBinColumns,
                                      nEvalVars, formula, nParms, columns, dtypes)
    wrapped_up = {}
    for key, val in temp.items():
        newkey = (key[0], 'jersf_lookup')
        vallist = list(val)
        vals, names = vallist[-1]
        names = ['central-up-down']
        central, down, up = vals
        vallist[-1] = (np.vstack((central.flatten(), up.flatten(), down.flatten())).T, names)
        wrapped_up[newkey] = tuple(vallist)

    return wrapped_up


def convert_junc_txt_file(juncFilePath):
    components = []
    basename = os.path.basename(juncFilePath).split('.')[0]
    with open(juncFilePath) as uncfile:
        for line in uncfile:
            if line.startswith('#'):
                continue
            elif line.startswith('['):
                component_name = line.strip()[1:-1]  # remove leading and trailing []
                cname = 'just/sum/dummy/dir/{0}_{1}.junc.txt'.format(basename, component_name)
                components.append((cname, []))
            elif components:
                components[-1][1].append(line)
            else:
                continue

    if not components:  # there are no components in the file
        components.append((juncFilePath, None))
    else:
        components = [(i, io.StringIO(''.join(j))) for i, j in components]

    retval = {}
    for name, ifile in components:
        retval.update(
            convert_junc_txt_component(name, ifile)
        )
    return retval


def convert_junc_txt_component(juncFilePath, uncFile):
    name, layout, pars, nBinnedVars, \
        nBinColumns, nEvalVars, formula, \
        nParms, columns, dtypes = _parse_jme_formatted_file(juncFilePath,
                                                            interpolatedFunc=True,
                                                            parmsFromColumns=True,
                                                            jme_f=uncFile)

    temp = _build_standard_jme_lookup(name, layout, pars, nBinnedVars, nBinColumns,
                                      nEvalVars, formula, nParms, columns, dtypes,
                                      interpolatedFunc=True)
    wrapped_up = {}
    for key, val in temp.items():
        newkey = (key[0], 'jec_uncertainty_lookup')
        vallist = list(val)
        vals, names = vallist[-1]
        knots = vals[0:len(vals):3]
        downs = vals[1:len(vals):3]
        ups = vals[2:len(vals):3]
        downs = np.array([down.flatten() for down in downs])
        ups = np.array([up.flatten() for up in ups])
        for knotv in knots:
            knot = np.unique(knotv.flatten())
            if knot.size != 1:
                raise Exception('Multiple bin low edges found')
        knots = np.array([np.unique(k.flatten())[0] for k in knots])
        vallist[2] = ({'knots': knots, 'ups': ups.T, 'downs': downs.T}, vallist[2][-1])
        vallist = vallist[:-1]
        wrapped_up[newkey] = tuple(vallist)
    return wrapped_up


def convert_effective_area_file(eaFilePath):
    ea_f = open(eaFilePath, 'r')
    layoutstr = ea_f.readline().strip().strip('{}')
    ea_f.close()

    name = eaFilePath.split('/')[-1].split('.')[0]

    layout = layoutstr.split()
    if not layout[0].isdigit():
        raise Exception('First column of Effective Area File Header must be a digit!')

    # setup the file format
    nBinnedVars = int(layout[0])
    nEvalVars = int(layout[nBinnedVars + 1])

    minMax = ['Min', 'Max']
    columns = []
    dtypes = []
    offset = 1
    for i in range(nBinnedVars):
        columns.extend(['%s%s' % (layout[i + offset], mm) for mm in minMax])
        dtypes.extend(['<f8', '<f8'])
    offset += nBinnedVars + 1
    for i in range(nEvalVars):
        columns.append('%s' % (layout[i + offset]))
        dtypes.append('<f8')

    pars = np.genfromtxt(eaFilePath,
                         dtype=tuple(dtypes),
                         names=tuple(columns),
                         skip_header=1,
                         unpack=True,
                         encoding='ascii'
                         )

    bins = {}
    offset_col = 0
    offset_name = 1
    bin_order = []
    for i in range(nBinnedVars):
        binMins = None
        binMaxs = None
        if i == 0:
            binMins = np.unique(pars[columns[0]])
            binMaxs = np.unique(pars[columns[1]])
            bins[layout[i + offset_name]] = np.union1d(binMins, binMaxs)
        else:
            counts = np.zeros(0, dtype=np.int)
            allBins = np.zeros(0, dtype=np.double)
            for binMin in bins[bin_order[0]][:-1]:
                binMins = np.unique(pars[np.where(pars[columns[0]] == binMin)][columns[i + offset_col]])
                binMaxs = np.unique(pars[np.where(pars[columns[0]] == binMin)][columns[i + offset_col + 1]])
                theBins = np.union1d(binMins, binMaxs)
                allBins = np.append(allBins, theBins)
                counts = np.append(counts, theBins.size)
            bins[layout[i + offset_name]] = awkward.JaggedArray.fromcounts(counts, allBins)
        bin_order.append(layout[i + offset_name])
        offset_col += 1

    # again this is only for one dimension of binning, fight me
    # we can figure out a 2D EA when we get there
    offset_name += 1
    wrapped_up = {}
    lookup_type = 'dense_lookup'
    dims = bins[layout[1]]
    for i in range(nEvalVars):
        ea_name = '_'.join([name, columns[offset_name + i]])
        values = pars[columns[offset_name + i]]
        wrapped_up[(ea_name, lookup_type)] = (values, dims)

    return wrapped_up
