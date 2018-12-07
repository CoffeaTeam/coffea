from __future__ import print_function

import numpy as np
from fnal_column_analysis_tools.lookup_tools.evaluator import evaluator

from fnal_column_analysis_tools.lookup_tools.root_converters import convert_histo_root_file
from fnal_column_analysis_tools.lookup_tools.csv_converters import convert_btag_csv_file
from fnal_column_analysis_tools.lookup_tools.json_converters import convert_histo_json_file

file_converters = {'root':convert_histo_root_file,
                   'csv':convert_btag_csv_file,
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

