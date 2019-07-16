from coffea import (hist,processor)

import warnings

import numpy as np

import pytest


def test_spark_imports():
    pyspark = pytest.importorskip("pyspark", minversion="2.4.1")
    
    from coffea.processor.spark.spark_executor import spark_executor
    from coffea.processor.spark.detail import (_spark_initialize,
                                               _spark_make_dfs,
                                               _spark_stop)

    spark = _spark_initialize()
    _spark_stop(spark)


def test_spark_executor():
    pyspark = pytest.importorskip("pyspark", minversion="2.4.1")
    from pyarrow.compat import guid
    
    from coffea.processor.spark.detail import (_spark_initialize,
                                               _spark_make_dfs,
                                               _spark_stop)
    from coffea.processor import run_spark_job

    import os
    import os.path as osp

    import pyspark.sql
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test-%s' % guid()) \
        .master('local[*]') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)

    spark = _spark_initialize(config=spark_config,log_level='ERROR',spark_progress=False)

    filelist = {'ZJets': ['file:'+osp.join(os.getcwd(),'tests/samples/nano_dy.root')],
                'Data'  : ['file:'+osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')]
                }

    from coffea.processor.test_items import NanoTestProcessor
    from coffea.processor.spark.spark_executor import spark_executor

    columns = ['nMuon','Muon_pt','Muon_eta','Muon_phi','Muon_mass']
    proc = NanoTestProcessor(columns=columns)

    hists = run_spark_job(filelist, processor_instance=proc, executor=spark_executor, spark=spark, thread_workers=1,
                          executor_args={'file_type': 'root'})

    _spark_stop(spark)

    assert( sum(spark_executor.counts.values()) == 20 )
    assert( hists['cutflow']['ZJets_pt'] == 4 )
    assert( hists['cutflow']['ZJets_mass'] == 1 )
    assert( hists['cutflow']['Data_pt'] == 15 )
    assert( hists['cutflow']['Data_mass'] == 5 )


def test_spark_hist_adders():
    pyspark = pytest.importorskip("pyspark", minversion="2.4.1")
    
    import pandas as pd
    import _pickle as pkl
    import lz4.frame as lz4f

    from coffea.util import numpy as np
    from coffea.processor.spark.spark_executor import agg_histos_raw, reduce_histos_raw
    from coffea.processor.test_items import NanoTestProcessor

    proc = NanoTestProcessor()

    one = proc.accumulator.identity()
    two = proc.accumulator.identity()
    hlist1 = [lz4f.compress(pkl.dumps(one))]
    hlist2 = [lz4f.compress(pkl.dumps(one)),lz4f.compress(pkl.dumps(two))]
    harray1 = np.array(hlist1, dtype='O')
    harray2 = np.array(hlist2, dtype='O')
    
    series1 = pd.Series(harray1)
    series2 = pd.Series(harray2)
    df = pd.DataFrame({'histos': harray2})

    # correctness of these functions is checked in test_spark_executor
    agg1 = agg_histos_raw(series1, proc, 1)
    agg2 = agg_histos_raw(series2, proc, 1)
    red = reduce_histos_raw(df, proc, 1)
