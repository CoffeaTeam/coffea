from fnal_column_analysis_tools.util import numpy as np
import json

def extract_json_histo_structure(parselevel,axis_names,axes):
    if 'value' in parselevel.keys():
        return
    name = list(parselevel)[0].split(':')[0]
    bins_pairs = [key.split(':')[-1].strip('[]').split(',') for key in parselevel.keys()]
    bins = []
    for pair in bins_pairs: bins.extend([float(val) for val in pair])
    bins.sort()
    bins = np.unique(np.array(bins))
    axis_names.append(name.encode())
    axes[axis_names[-1]] = bins
    extract_json_histo_structure(parselevel[list(parselevel)[0]],axis_names,axes)

def extract_json_histo_values(parselevel,binlows,values,val_names):
    if 'value' in parselevel.keys():
        binvals = {}
        binvals.update(parselevel)
        keylows = tuple(binlows)
        values[keylows] = binvals
        for key in parselevel.keys():
            val_names.add(key)
        return
    for key in parselevel.keys():
        lowside = float(key.split(':')[-1].strip('[]').split(',')[0])
        thelows = [lowside]
        if len(binlows) != 0: thelows = binlows + thelows
        extract_json_histo_values(parselevel[key],thelows,values,val_names)

def convert_histo_json_file(filename):
    file = open(filename)
    info = json.load(file)
    file.close()
    names_and_orders = {}
    names_and_axes = {}
    names_and_binvalues = {}
    names_and_valnames = {}
    
    #first pass, convert info['dir']['hist_title'] to dir/hist_title
    #and un-nest everything from the json structure, make binnings, etc.
    for dir in info.keys():
        for htitle in info[dir].keys():
            axis_order = [] #keep the axis order
            axes = {}
            bins_and_values = {}
            val_names = set()
            extract_json_histo_structure(info[dir][htitle],axis_order,axes)
            extract_json_histo_values(info[dir][htitle],[],bins_and_values,val_names)
            histname = '%s/%s'%(dir,htitle)
            names_and_axes[histname] = axes
            names_and_orders[histname] = axis_order
            names_and_binvalues[histname] = bins_and_values
            names_and_valnames[histname] = val_names
    
    wrapped_up = {}
    for name,axes in names_and_axes.items():
        theshape = tuple([axes[axis].size-1 for axis in names_and_orders[name]])
        valsdict = {}
        for vname in names_and_valnames[histname]:
            valsdict[vname] = np.zeros(shape=theshape).flatten()
        flatidx = np.arange(np.zeros(shape=theshape).size)
        binidx = np.unravel_index(flatidx, shape=theshape)
        for vname in valsdict:
            for iflat in flatidx:
                binlows = []
                for idim,axis in enumerate(names_and_orders[name]):
                    binlows.append(axes[axis][binidx[idim][iflat]])
                thevals = names_and_binvalues[name][tuple(binlows)]
                valsdict[vname][iflat] = thevals[vname]
            valsdict[vname] = valsdict[vname].reshape(theshape)
        bins_in_order = []
        for axis in names_and_orders[name]:
            bins_in_order.append(axes[axis])
        for vname in valsdict:
            wrapped_up[(name+'_'+vname,'dense_lookup')] = (valsdict[vname],tuple(bins_in_order))
    return wrapped_up
