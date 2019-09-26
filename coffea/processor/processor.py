from six import with_metaclass
from abc import ABCMeta, abstractmethod


class ProcessorABC(with_metaclass(ABCMeta)):
    '''ABC for a generalized processor

    The various data delivery mechanisms (spark, striped, uproot, uproot+futures, condor, ...)
    receive such an object and the appropriate metadata to deliver dataframes to it.
    It is expected that the entire processor object can be serializable (check with `coffea.util.save`)
    No attempt should be made to track state inside an instance of ProcessorABC, it is to be
    treated simply as a bundle of methods.  The only exception is the read-only accumulator
    property.
    '''
    @property
    @abstractmethod
    def accumulator(self):
        '''Read-only accumulator object

        It is up to the derived class to define this, in e.g. the initializer.
        '''
        pass

    @abstractmethod
    def process(self, df):
        '''Processes a single DataFrame

        Returns a filled accumulator object which can be initialized via self.accumulator.identity()
        '''
        pass

    @abstractmethod
    def postprocess(self, accumulator):
        '''Final processing on aggregated accumulator

        Do any final processing on the resulting accumulator object, and return it
        '''
        pass
