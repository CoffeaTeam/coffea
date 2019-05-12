def parsl_executor(config, items, function, accumulator, workers, status, unit, desc):
    # create dataflow kernel

    # submit jobs

    # collect them

    raise NotImplementedError
    return accumulator


default_condor_cfg = None


def condor_executor(items, function, accumulator, workers, status=True, unit='items', desc='Processing', cfg=default_condor_cfg):
    return parsl_executor(cfg, items, function, accumulator, workers, status, unit, desc)


default_slurm_cfg = None


def slurm_executor(items, function, accumulator, workers, status=True, unit='items', desc='Processing', cfg=default_slurm_cfg):
    return parsl_executor(cfg, items, function, accumulator, workers, status, unit, desc)
