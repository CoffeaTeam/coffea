import numpy as np
import awkward
from awkward import JaggedArray

#for later
#func = numbaize(formula,['p%i'%i for i in range(nParms)]+[varnames[i] for i in range(nEvalVars)])

def convert_jec_txt_file(jecFilePath):
    jec_f = open(jecFilePath,'r')
    layoutstr = jec_f.readline().strip().strip('{}')
    jec_f.close()

    name = jecFilePath.split('/')[-1].split('.')[0]
    
    layout = layoutstr.split()
    if not layout[0].isdigit():
        raise Exception('First column of JEC descriptor must be a digit!')

    #setup the file format
    nBinnedVars = int(layout[0])
    nBinColumns = 2*nBinnedVars
    nEvalVars   = int(layout[nBinnedVars+1])
    formula     = layout[nBinnedVars+nEvalVars+2]
    nParms      = 0
    while( formula.count('[%i]'%nParms) ):
        formula = formula.replace('[%i]'%nParms,'p%i'%nParms)
        nParms += 1
    #protect function names with vars in them
    funcs_to_cap = ['max','exp']
    for f in funcs_to_cap:
        formula = formula.replace(f,f.upper())

    templatevars = ['x','y','z','w','t','s']
    varnames = [layout[i+nBinnedVars+2] for i in range(nEvalVars)]
    for find,replace in zip(templatevars,varnames):
        formula = formula.replace(find,replace)
    #restore max
    for f in funcs_to_cap:
        formula = formula.replace(f.upper(),f)
    nFuncColumns = 2*nEvalVars + nParms
    nTotColumns = nFuncColumns + 1

    #parse the columns
    minMax = ['Min','Max']
    columns = []
    dtypes = []
    offset = 1
    for i in range(nBinnedVars):
        columns.extend(['%s%s'%(layout[i+offset],mm) for mm in minMax])
        dtypes.extend(['<f8','<f8'])
    columns.append('NVars')
    dtypes.append('<i8')
    offset += nBinnedVars + 1
    for i in range(nEvalVars):
        columns.extend(['%s%s'%(layout[i+offset],mm) for mm in minMax])
        dtypes.extend(['<f8','<f8'])
    for i in range(nParms):
        columns.append('p%i'%i)
        dtypes.append('<f8')

    pars = np.genfromtxt(jecFilePath,
                         dtype=tuple(dtypes),
                         names=tuple(columns),
                         skip_header=1,
                         unpack=True,
                         encoding='ascii'
                         )

    #the first bin is always usual for JECs
    #the next bins may vary in number, so they're jagged arrays... yay
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
            bins[layout[i+offset_name]] = np.union1d(binMins,binMaxs)
        else:
            counts = np.zeros(0,dtype=np.int)
            allBins = np.zeros(0,dtype=np.double)
            for binMin in bins[bin_order[0]][:-1]:
                binMins = np.unique(pars[np.where(pars[columns[0]] == binMin)][columns[i+offset_col]])
                binMaxs = np.unique(pars[np.where(pars[columns[0]] == binMin)][columns[i+offset_col+1]])
                theBins = np.union1d(binMins,binMaxs)
                allBins = np.append(allBins,theBins)
                counts  = np.append(counts,theBins.size)
            bins[layout[i+offset_name]] = JaggedArray.fromcounts(counts,allBins)
        bin_order.append(layout[i+offset_name])
        offset_col += 1

    #skip nvars to the variable columns
    #the columns here define clamps for the variables defined in columns[]
    # ----> clamps can be different from bins
    # ----> if there is more than one binning variable this array is jagged
    # ----> just make it jagged all the time
    binshapes = tuple([bins[thebin].size-1 for thebin in bin_order])
    clamp_mins = {}
    clamp_maxs = {}
    var_order = []
    offset_col = 2*nBinnedVars+1
    offset_name = nBinnedVars + 2
    jagged_counts = np.ones(bins[bin_order[0]].size-1,dtype=np.int)
    if len(bin_order) > 1:
        jagged_counts = np.maximum(bins[bin_order[1]].counts - 1,0) #need counts-1 since we only care about Nbins
    for i in range(nEvalVars):
        clamp_mins[layout[i+offset_name]] = JaggedArray.fromcounts(jagged_counts,np.atleast_1d(pars[columns[i+offset_col]]))
        clamp_maxs[layout[i+offset_name]] = JaggedArray.fromcounts(jagged_counts,np.atleast_1d(pars[columns[i+offset_col+1]]))
        var_order.append(layout[i+offset_name])
        offset_col += 1

    #now get the parameters, which we will look up with the clamps
    parms = []
    parm_order = []
    offset_col = 2*nBinnedVars+1 + 2*nEvalVars
    for i in range(nParms):
        parms.append(JaggedArray.fromcounts(jagged_counts,pars[columns[i+offset_col]]))
        parm_order.append('p%i'%(i))
    
    wrapped_up = {}
    wrapped_up[(name,'jet_energy_corrector')] = (formula,
                                                 (bins,bin_order),
                                                 (clamp_mins,clamp_maxs,var_order),
                                                 (parms,parm_order))
    return wrapped_up
