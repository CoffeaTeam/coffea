from fnal_column_analysis_tools.lookup_tools.dense_lookup import dense_lookup
from fnal_column_analysis_tools.lookup_tools.dense_evaluated_lookup import dense_evaluated_lookup
from fnal_column_analysis_tools.lookup_tools.jet_energy_corrector import jet_energy_corrector

lookup_types = {'dense_lookup':dense_lookup,
                'dense_evaluated_lookup':dense_evaluated_lookup,
                'jet_energy_corrector':jet_energy_corrector
               }

class evaluator(object):
    def __init__(self,names,types,primitives):
        self._functions = {}
        for key in names.keys():
            lookup_type = types[names[key]]
            lookup_def = primitives[names[key]]
            self._functions[key] = lookup_types[lookup_type](*lookup_def)
            
    def __dir__(self):
        return self._functions.keys()
        
    def __getitem__(self, key):
        return self._functions[key]
