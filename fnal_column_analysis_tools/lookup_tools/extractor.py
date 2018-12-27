from __future__ import print_function

import numpy as np
from .evaluator import evaluator

from .root_converters import convert_histo_root_file
from .csv_converters import convert_btag_csv_file
from .json_converters import convert_histo_json_file
from .txt_converters import *

file_converters = {'root':{'default':convert_histo_root_file,
                           'histo':convert_histo_root_file},
                   'csv':{'default':convert_btag_csv_file,
                          'btag':convert_btag_csv_file},
                   'json':{'default':convert_histo_json_file,
                           'histo':convert_histo_json_file},
                   'txt':{'default':convert_jec_txt_file,
                          'jec':convert_jec_txt_file,
                          'jersf':convert_jersf_txt_file,
                          'jr':convert_jr_txt_file,
                          'junc':convert_junc_txt_file,
                          'ea':convert_effective_area_file}
                   }

class extractor(object):
    def __init__(self):
        self._weights = []
        self._names = {}
        self._types = []
        self._filecache = {}
        self._finalized = False
    
    def add_weight_set(self,local_name,type,weights):
        if self._finalized:
            raise Exception('extractor is finalized cannot add new weights!')
        if local_name in self._names.keys():
            raise Exception('weights name "{}" already defined'.format(local_name))
        self._names[local_name] = len(self._weights)
        self._types.append(type)
        self._weights.append(weights)
    
    def add_weight_sets(self,weightsdescs):
        # expect file to be formatted <local name> <name> <weights file>
        # allow * * <file> and <prefix> * <file> to do easy imports of whole file
        for weightdesc in weightsdescs:
            if weightdesc[0] == '#': continue #skip comment lines
            temp = weightdesc.strip().split(' ')
            if len(temp) != 3: 
                raise Exception('"{}" not formatted as "<local name> <name> <weights file>"'.format(weightdesc))
            (local_name,name,file) = tuple(temp)
            if name == '*':
                self.import_file(file)
                weights = self._filecache[file]
                for key, value in weights.items():
                    if local_name == '*':
                        self.add_weight_set(key[0],key[1],value)
                    else:
                        self.add_weight_set(local_name+key[0],key[1],value)
            else:
                weights,type = self.extract_from_file(file,name)
                self.add_weight_set(local_name,type,weights)
    
    def import_file(self,file):
        if file not in self._filecache.keys():
            file_dots = file.split('.')
            format = file_dots[-1].strip()
            type = 'default'
            if len(file_dots) > 2:
                type = file_dots[-2]
            self._filecache[file] = file_converters[format][type](file)
    
    def extract_from_file(self, file, name):        
        self.import_file(file)
        weights = self._filecache[file]
        names = {key[0]:key[1] for key in weights.keys()}
        bname = name.encode()
        if bname not in names.keys():
            raise Exception('Weights named "{}" not in {}!'.format(name,file))
        return (weights[(bname,names[bname])],names[bname])
          
    def finalize(self,reduce_list=None):
        if self._finalized:
            raise Exception('extractor is already finalized!')
        del self._filecache
        if reduce_list is not None:
            names = {}
            types = []
            weights = []
            for i,name in enumerate(reduce_list):
                if name not in self._names:
                    raise Exception('Weights named "{}" not in extractor!'.format(name))
                names[name] = i
                types.append(self._types[self._names[name]])
                weights.append(self._weights[self._names[name]])
            self._names = names
            self._types = types
            self._weights = weights
        self._finalized = True
    
    def make_evaluator(self):
        return evaluator(self._names,
                         self._types,
                         self._weights)

