from coffea import (hist,processor)

import warnings

import numpy as np

import pytest
import sys

if sys.version.startswith("3.8"):
    pytest.skip("pyspark not yet functional in python 3.8", allow_module_level=True)

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
        .config('spark.executor.x509proxyname','x509_u12409') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)

    spark = _spark_initialize(config=spark_config,log_level='ERROR',spark_progress=False)

    filelist = {'ZJets': {'files': ['file:'+osp.join(os.getcwd(),'tests/samples/nano_dy.root')], 'treename': 'Events' },
                'Data'  : {'files': ['file:'+osp.join(os.getcwd(),'tests/samples/nano_dimuon.root')], 'treename': 'Events'}
                }

    from coffea.processor.test_items import NanoTestProcessor, NanoEventsProcessor
    from coffea.processor.spark.spark_executor import spark_executor

    columns = ['nMuon','Muon_pt','Muon_eta','Muon_phi','Muon_mass', 'Muon_charge']
    proc = NanoTestProcessor(columns=columns)

    hists = run_spark_job(filelist, processor_instance=proc, executor=spark_executor, spark=spark, thread_workers=1,
                          executor_args={'file_type': 'root'})

    assert( sum(spark_executor.counts.values()) == 80 )
    assert( hists['cutflow']['ZJets_pt'] == 18 )
    assert( hists['cutflow']['ZJets_mass'] == 6 )
    assert( hists['cutflow']['Data_pt'] == 84 )
    assert( hists['cutflow']['Data_mass'] == 66 )

    hists = run_spark_job(filelist, processor_instance=proc, executor=spark_executor, spark=spark, thread_workers=1,
                          executor_args={'file_type': 'root', 'flatten': True})
    
    assert( sum(spark_executor.counts.values()) == 80 )
    assert( hists['cutflow']['ZJets_pt'] == 18 )
    assert( hists['cutflow']['ZJets_mass'] == 6 )
    assert( hists['cutflow']['Data_pt'] == 84 )
    assert( hists['cutflow']['Data_mass'] == 66 )
    
    proc = NanoEventsProcessor(columns=columns)
    hists = run_spark_job(filelist, processor_instance=proc, executor=spark_executor, spark=spark, thread_workers=1,
                          executor_args={'file_type': 'root', 'nano': True})

    _spark_stop(spark)

    assert( sum(spark_executor.counts.values()) == 80 )
    assert( hists['cutflow']['ZJets_pt'] == 18 )
    assert( hists['cutflow']['ZJets_mass'] == 6 )
    assert( hists['cutflow']['Data_pt'] == 84 )
    assert( hists['cutflow']['Data_mass'] == 66 )


def test_spark_hist_adders():
    pyspark = pytest.importorskip("pyspark", minversion="2.4.1")
    
    import pandas as pd
    import pickle as pkl
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
