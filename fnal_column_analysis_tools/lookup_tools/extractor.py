from __future__ import print_function

import re
import uproot
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

file_converters = {'root':convert_root_file,
                   'csv':convert_btag_csv}

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

