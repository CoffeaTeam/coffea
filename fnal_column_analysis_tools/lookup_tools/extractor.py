from __future__ import print_function

import re
import uproot
import json
import numpy as np
from fnal_column_analysis_tools.lookup_tools.evaluator import evaluator

TH1D = "<class 'uproot.rootio.TH1D'>"
TH2D = "<class 'uproot.rootio.TH2D'>"
TH1F = "<class 'uproot.rootio.TH1F'>"
TH2F = "<class 'uproot.rootio.TH2F'>"
RootDir = "<class 'uproot.rootio.ROOTDirectory'>"

def convert_root_file(file):
    converted_file = {}
    dumFile = uproot.open(file.strip())
    for i, key in enumerate(dumFile.keys()):
        histType = str(type(dumFile[key[:-2]]))
        if histType == TH1D or histType == TH2D or histType==TH1F or histType==TH2F:
            if  not ("bound method" in str(dumFile[key[:-2]].numpy)):
                converted_file[key[:-2]] = dumFile[key[:-2]].numpy
            else:
                converted_file[key[:-2]] = dumFile[key[:-2]].numpy()
        elif histType == RootDir: #means there are subdirectories wihin main directory
            for j, key2 in enumerate(dumFile[key[:-2]].keys()):
                histType2 = str(type(dumFile[key[:-2]][key2[:-2]]))
                if histType2 == TH1D or histType2 == TH2D or histType2==TH1F or histTyp2==TH2F:
                    if not ("bound method" in str(dumFile[key[:-2]][key2[:-2]].numpy)):
                        converted_file[key[:-2]+'/'+key2[:-2]] = dumFile[key[:-2]][key2[:-2]].numpy
                    else:
                        converted_file[key[:-2]+'/'+key2[:-2]] = dumFile[key[:-2]][key2[:-2]].numpy()           
        else:
            tempArrX= dumFile[key[:-2]]._fEX
            tempArrY= dumFile[key[:-2]]._fEY
            converted_file[key[:-2]] = [tempArrX, tempArrY]
    return converted_file

#pt except for reshaping, then discriminant
btag_feval_dims = {0:[1],1:[1],2:[1],3:[2]}

def convert_btag_csv(csvFilePath):
    f = open(csvFilePath).readlines()
    columns = f.pop(0)
    nameandcols = columns.split(';')
    name = nameandcols[0].strip()
    columns = nameandcols[1].strip()
    columns = [column.strip() for column in columns.split(',')]
    
    corrections = np.genfromtxt(csvFilePath,
                                dtype=None,
                                names=tuple(columns),
                                converters={1:lambda s: s.strip(),
                                            2:lambda s: s.strip(),
                                           10:lambda s: s.strip(' "')},
                                delimiter = ',',
                                skip_header=1,
                                unpack=True,
                                encoding='ascii'
                                )
    
    all_names = corrections[[columns[i] for i in range(4)]]
    labels = np.unique(corrections[[columns[i] for i in range(4)]])
    names_and_bins = np.unique(corrections[[columns[i] for i in [0,1,2,3,4,6,8]]])
    wrapped_up = {}
    for label in labels:
        etaMins = np.unique(corrections[np.where(all_names == label)][columns[4]])
        etaMaxs = np.unique(corrections[np.where(all_names == label)][columns[5]])
        etaBins = np.union1d(etaMins,etaMaxs)
        ptMins = np.unique(corrections[np.where(all_names == label)][columns[6]])
        ptMaxs = np.unique(corrections[np.where(all_names == label)][columns[7]])
        ptBins = np.union1d(ptMins,ptMaxs)
        discrMins = np.unique(corrections[np.where(all_names == label)][columns[8]])
        discrMaxs = np.unique(corrections[np.where(all_names == label)][columns[9]])
        discrBins = np.union1d(discrMins,discrMaxs)
        vals = np.zeros(shape=(len(discrBins)-1,len(ptBins)-1,len(etaBins)-1),
                        dtype=corrections.dtype[10])
        for i,eta_bin in enumerate(etaBins[:-1]):
            for j,pt_bin in enumerate(ptBins[:-1]):
                for k,discr_bin in enumerate(discrBins[:-1]):
                    this_bin = np.where((all_names == label) &
                                        (corrections[columns[4]] == eta_bin) &
                                        (corrections[columns[6]] == pt_bin)  &
                                        (corrections[columns[8]] == discr_bin))
                    vals[k,j,i] = corrections[this_bin][columns[10]][0]
        label_decode = []
        for i in range(len(label)):
            label_decode.append(label[i])
            if isinstance(label_decode[i],bytes):
                label_decode[i] = label_decode[i].decode()
            else:
                label_decode[i] = str(label_decode[i])
        str_label = '_'.join([name]+label_decode)
        feval_dim = btag_feval_dims[label[0]]
        wrapped_up[str_label] = (vals,(etaBins,ptBins,discrBins),tuple(feval_dim))
    return wrapped_up

def extract_json_histo_structure(parselevel,axis_names,axes):
    if 'value' in parselevel.keys():
        return
    name = parselevel.keys()[0].split(':')[0]
    bins_pairs = [key.split(':')[-1].strip('[]').split(',') for key in parselevel.keys()]
    bins = []
    for pair in bins_pairs: bins.extend([float(val) for val in pair])
    bins.sort()
    bins = np.unique(np.array(bins))
    axis_names.append(name.encode())
    axes[axis_names[-1]] = bins
    extract_json_histo_structure(parselevel[parselevel.keys()[0]],axis_names,axes)

def extract_json_histo_values(parselevel,binlows,values):
    if 'value' in parselevel.keys():
        binvals = {}
        binvals.update(parselevel)
        keylows = tuple(binlows)
        values[keylows] = binvals
        return
    for key in parselevel.keys():
        lowside = float(key.split(':')[-1].strip('[]').split(',')[0])
        thelows = [lowside]
        if len(binlows) != 0: thelows = binlows + thelows
        extract_json_histo_values(parselevel[key],thelows,values)

#TODO: make it work for an arbitrary number of values/errors/whatever
def convert_histo_json_file(filename):
    file = open(filename)
    info = json.load(file)
    file.close()
    names_and_orders = {}
    names_and_axes = {}
    names_and_binvalues = {}
    
    #first pass, convert info['dir']['hist_title'] to dir/hist_title
    #and un-nest everything from the json structure, make binnings, etc.
    for dir in info.keys():
        for htitle in info[dir].keys():
            axis_order = [] #keep the axis order
            axes = {}
            bins_and_values = {}
            extract_json_histo_structure(info[dir][htitle],axis_order,axes)
            extract_json_histo_values(info[dir][htitle],[],bins_and_values)
            histname = '%s/%s'%(dir.encode(),htitle.encode())
            names_and_axes[histname] = axes
            names_and_orders[histname] = axis_order
            names_and_binvalues[histname] = bins_and_values
    
    wrapped_up = {}
    for name,axes in names_and_axes.items():
        theshape = tuple([axes[axis].size-1 for axis in names_and_orders[name]])
        values = np.zeros(shape=theshape).flatten()
        errors = np.zeros(shape=theshape).flatten()
        flatidx = np.arange(values.size)
        binidx = np.unravel_index(flatidx,dims=theshape)
        for iflat in flatidx:
            binlows = []
            for idim,axis in enumerate(names_and_orders[name]):
                binlows.append(axes[axis][binidx[idim][iflat]])
            thevals = names_and_binvalues[name][tuple(binlows)]
            values[iflat] = thevals['value']
            errors[iflat] = thevals['error']
        values = values.reshape(theshape)
        errors = errors.reshape(theshape)
        bins_in_order = []
        for axis in reversed(names_and_orders[name]):
            bins_in_order.append(axes[axis])
        wrapped_up[name] = (values,tuple(bins_in_order))
        wrapped_up[name+'_err'] = (errors,tuple(bins_in_order))
    return wrapped_up

file_converters = {'root':convert_root_file,
                   'csv':convert_btag_csv,
                   'json':convert_histo_json_file}

class extractor(object):
    def __init__(self):
        self._weights = []
        self._names = {}
        self._filecache = {}
        self._finalized = False
    
    def add_weight_set(self,local_name,weights):
        if self._finalized:
            raise Exception('extractor is finalized cannot add new weights!')
        if local_name in self._names.keys():
            raise Exception('weights name "{}" already defined'.format(local_name))
        self._names[local_name] = len(self._weights)
        self._weights.append(weights)
    
    def add_weight_sets(self,weightsdescs):
        # expect file to be formatted <local name> <name> <weights file>
        # allow * * <file> and <prefix> * <file> to do easy imports of whole file
        for weightdesc in weightsdescs:
            temp = weightdesc.split(' ')
            if len(temp) != 3: 
                raise Exception('"{}" not formatted as "<local name> <name> <weights file>"'.format(weightdesc))
            (local_name,name,file) = tuple(temp)
            if name == '*':
                self.import_file(file)
                weights = self._filecache[file]
                for key, value in weights.items():
                    if local_name == '*':
                        self.add_weight_set(key,value)
                    else:
                        self.add_weight_set(local_name+key,value)
            else:
                weights = self.extract_from_file(file,name)
                self.add_weight_set(local_name,weights)
    
    def import_file(self,file):
        if file not in self._filecache.keys():
            self._filecache[file] = file_converters[file.split('.')[-1].strip()](file)
    
    def extract_from_file(self, file, name):        
        self.import_file(file)
        weights = self._filecache[file]
        bname = name.encode()
        if bname not in weights.keys(): 
            print(weights.keys())
            raise Exception('Weights named "{}" not in {}!'.format(name,file))        
        return weights[bname]
          
    def finalize(self):
        if self._finalized:
            raise Exception('extractor is already finalized!')
        del self._filecache
        self._finalized = True
    
    def make_evaluator(self,reduce_list=None):
        names = None
        weights = None
        if reduce_list is not None:
            names = {}
            weights = []
            for i,name in enumerate(reduce_list):
                if name not in self._names:
                    raise Exception('Weights named "{}" not in extractor!'.format(name))
                names[name] = i
                weights.append(self._weights[self._names[name]])
        else:
            names = self._names
            weights = self._weights
        
        return evaluator(names,weights)

