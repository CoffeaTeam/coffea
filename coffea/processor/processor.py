from abc import ABCMeta, abstractmethod


class ProcessorABC(metaclass=ABCMeta):
    '''ABC for a generalized processor

    The various data delivery mechanisms (spark, striped, uproot, uproot+futures, condor, ...)
    receive such an object and the appropriate metadata to deliver dataframes to it.
    It is expected that the entire processor object can be serializable (check with `coffea.util.save`)
    No attempt should be made to track state inside an instance of ProcessorABC, it is to be
    treated simply as a bundle of methods.

    Examples
    --------

    A skeleton processor::

        from collections import defaultdict
        from coffea import hist, processor

        class MyProcessor(processor.ProcessorABC):
            def __init__(self, flag=False):
                self._flag = flag
                    "sumw": processor.defaultdict_accumulator(float),
                })

            def process(self, events):
                out = {"sumw": defaultdict(int)}
                out["sumw"]["this dataset"] += 20.

                # ...

                return out

            def postprocess(self, accumulator):
                return accumulator

        p = MyProcessor()

    '''
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
