from .dense_lookup import dense_lookup
from .dense_evaluated_lookup import dense_evaluated_lookup
from .jme_standard_function import jme_standard_function
from .jersf_lookup import jersf_lookup
from .jec_uncertainty_lookup import jec_uncertainty_lookup

lookup_types = {'dense_lookup': dense_lookup,
                'dense_evaluated_lookup': dense_evaluated_lookup,
                'jme_standard_function': jme_standard_function,
                'jersf_lookup': jersf_lookup,
                'jec_uncertainty_lookup': jec_uncertainty_lookup
               }


class evaluator(object):
    """
        The evaluator class serves as a single point of extry for
        looking up values of histograms and other functions read in
        with the extractor class. Stored look ups can be indexed by
        name and then called through an overloaded __call__ function.
        Example:
            evaluate = extractor.make_evaluator()
            vals = evaluate[XYZ](arg1,arg2,...)
        The returned 'vals' has the same shape as the input args.

        lookup_types is a map of possible contructors for extracted data
    """
    def __init__(self, names, types, primitives):
        """
            initialize the evaluator from a list of inputs names,
            lookup types, and input primitive data
        """
        self._functions = {}
        for key in names.keys():
            lookup_type = types[names[key]]
            lookup_def = primitives[names[key]]
            self._functions[key] = lookup_types[lookup_type](*lookup_def)

    def __dir__(self):
        """ dir is overloaded to list all available functions
            in the evaluator
        """
        return self._functions.keys()

    def __getitem__(self, key):
        """ return a function named 'key' """
        return self._functions[key]

    def keys(self):
        """ returns the available functions """
        return self._functions.keys()

    def __contains__(self, item):
        """ item in X """
        return item in self._functions
