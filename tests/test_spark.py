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


@pytest.mark.skip(reason='test should work but does not.')
def test_spark_functionality():
    pyspark = pytest.importorskip("pyspark", minversion="2.4.1")
    
    import lz4.frame as lz4f
    import cloudpickle as cpkl
    
    from coffea import hist
    from coffea.processor.spark.tests import check_spark_functionality
    one,two,hists = check_spark_functionality()

    #check that spark produced the correct output dataset
    assert( (one == two) and (one == 3) )

    #make sure that the returned histograms are all the same (as they should be in this case)
    assert( (len(hists[0]) == len(hists[1])) and (len(hists[1]) == len(hists[2])) )

    inflated = []
    for i in range(3):
        inflated.append(cpkl.loads(lz4f.decompress(hists[i])))

    final = inflated.pop()
    for ihist in inflated:
        final.add(ihist)

    #make sure the accumulator is working right after being blobbed through spark
    assert( final['cutflow']['dummy'] == 3 )

def test_spark_executor():
    pyspark = pytest.importorskip("pyspark", minversion="2.4.1")
    
    from coffea.processor.spark.detail import (_spark_initialize,
                                               _spark_make_dfs,
                                               _spark_stop)
    from coffea.processor import run_spark_job

    import os
    import os.path as osp

    import pyspark.sql
    spark_config = pyspark.sql.SparkSession.builder \
        .appName('spark-executor-test') \
        .master('local[*]') \
        .config('spark.sql.execution.arrow.enabled','true') \
        .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)

    spark = _spark_initialize(config=spark_config,log_level='ERROR',spark_progress=False)

    filelist = {'ZJets': ['file:'+osp.join(os.getcwd(),'tests/samples/nano_dy.parquet')],
                'Data'  : ['file:'+osp.join(os.getcwd(),'tests/samples/nano_dimuon.parquet')]
                }

    from coffea.processor.test_items import NanoTestProcessor
    from coffea.processor.spark.spark_executor import spark_executor

    columns = ['nMuon','Muon_pt','Muon_eta','Muon_phi','Muon_mass']
    proc = NanoTestProcessor(columns=columns)

    hists = run_spark_job(filelist, processor_instance=proc, executor=spark_executor, spark=spark, thread_workers=1)

    _spark_stop(spark)

    assert( sum(spark_executor.counts.values()) == 20 )
    assert( hists['cutflow']['ZJets_pt'] == 4 )
    assert( hists['cutflow']['ZJets_mass'] == 1 )
    assert( hists['cutflow']['Data_pt'] == 15 )
    assert( hists['cutflow']['Data_mass'] == 5 )

