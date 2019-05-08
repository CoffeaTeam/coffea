import parsl
from parsl.app.app import python_app, bash_app

def parsl_base_executor(config, items, function, accumulator, workers, status, unit, desc):
    #create dfk

    #prepare the processor

    #submit jobs

    #collect them

    #reset?

    #kill the dfk?

    raise NotImplementedError
    return accumulator

default_condor_cfg = None

def condor_executor(items, function, accumulator, workers, status=True, unit='items', desc='Processing', cfg=default_condor_cfg):
    
    return parsl_executor(cfg, items, function, accumulator, workers, status, unit, desc)

def slurm_executor(items, function, accumulator, workers, status=True, unit='items', desc='Processing', cfg=default_slurm_cfg):
    config = None
    return 
