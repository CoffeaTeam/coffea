from .dataframe import DataFrame
from .helpers import Weights, PackedSelection


class ProcessorBase(object):
    """
    Strawman for a generalized processor
    The various data delivery mechanisms (spark, striped, local uproot, multiprocessing uproot, condor, ...)
    could receive such an object and the appropriate metadata to deliver dataframes to it.
    """
    def __init__(self):
        """
        Initialize with:
            all accumulator definitions (preferably constructed empty Hist objects)
            all external data (evaluator instance or dictionary, or whatever)
        These objects should be cloudpickle-able, and read-only (or evaluate-only in the case of
        function definitions) TODO: enforce by special class?
        Copies of accumulator definitions must be made via accumulators() for writing
        """
        raise NotImplementedError

    def accumulators(self):
        """
        Construct (by copy or instantiation from definition) writeable clones
        of all accumulators.  The clones should start empty.
        The clones could be set as member variables if desired, but cannot overwrite
        the read-only accumulators that were initially instantiated.
        """
        raise NotImplementedError

    def process(self, df):
        """
        Processes a single DataFrame
        Returns arbitrary output, to be handled by collect() on the originating executable
        """
        raise NotImplementedError

    def collect(self, output):
        """
        Receiver of output from (possibly many distributed) process() calls.
        Again, some writeable copy of accumulators should be made, and stored
        as a member variable.
        """
        raise NotImplementedError
