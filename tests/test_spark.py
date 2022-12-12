import pytest


def test_spark_imports():
    pytest.importorskip("pyspark", minversion="3.3.0")

    from coffea.processor.spark.detail import _spark_initialize, _spark_stop

    spark = _spark_initialize(bindAddress="127.0.0.1", host="127.0.0.1")
    _spark_stop(spark)


@pytest.mark.skip(reason="pyspark executor work currently in progress")
def test_spark_executor():
    pyspark = pytest.importorskip("pyspark", minversion="3.3.0")
    import os
    import os.path as osp

    import pyspark.sql
    from pyarrow.util import guid

    from coffea.nanoevents import schemas
    from coffea.processor import run_spark_job
    from coffea.processor.spark.detail import _spark_initialize, _spark_stop

    spark_config = (
        pyspark.sql.SparkSession.builder.appName("spark-executor-test-%s" % guid())
        .master("local[*]")
        .config("spark.sql.execution.arrow.enabled", "true")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.executor.x509proxyname", "x509_u12409")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", 200000)
    )

    spark = _spark_initialize(
        config=spark_config, log_level="ERROR", spark_progress=False
    )

    filelist = {
        "ZJets": {
            "files": ["file:" + osp.join(os.getcwd(), "tests/samples/nano_dy.root")],
            "treename": "Events",
        },
        "Data": {
            "files": [
                "file:" + osp.join(os.getcwd(), "tests/samples/nano_dimuon.root")
            ],
            "treename": "Events",
        },
    }

    from coffea.processor.spark.spark_executor import spark_executor
    from coffea.processor.test_items import NanoEventsProcessor, NanoTestProcessor

    columns = ["nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge"]
    proc = NanoTestProcessor(columns=columns)

    hists = run_spark_job(
        filelist,
        processor_instance=proc,
        executor=spark_executor,
        spark=spark,
        thread_workers=1,
        executor_args={"file_type": "root"},
    )

    assert sum(spark_executor.counts.values()) == 80
    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66

    hists = run_spark_job(
        filelist,
        processor_instance=proc,
        executor=spark_executor,
        spark=spark,
        thread_workers=1,
        executor_args={"file_type": "root"},
    )

    assert sum(spark_executor.counts.values()) == 80
    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66

    proc = NanoEventsProcessor(columns=columns)
    hists = run_spark_job(
        filelist,
        processor_instance=proc,
        executor=spark_executor,
        spark=spark,
        thread_workers=1,
        executor_args={"file_type": "root", "schema": schemas.NanoAODSchema},
    )

    _spark_stop(spark)

    assert sum(spark_executor.counts.values()) == 80
    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66


def test_spark_hist_adders():
    pytest.importorskip("pyspark", minversion="3.3.0")

    import pickle as pkl

    import lz4.frame as lz4f
    import pandas as pd

    from coffea.processor.spark.spark_executor import agg_histos_raw, reduce_histos_raw
    from coffea.processor.test_items import NanoTestProcessor
    from coffea.util import numpy as np

    proc = NanoTestProcessor()

    one = proc.accumulator
    two = proc.accumulator
    hlist1 = [lz4f.compress(pkl.dumps(one))]
    hlist2 = [lz4f.compress(pkl.dumps(one)), lz4f.compress(pkl.dumps(two))]
    harray1 = np.array(hlist1, dtype="O")
    harray2 = np.array(hlist2, dtype="O")

    series1 = pd.Series(harray1)
    series2 = pd.Series(harray2)
    df = pd.DataFrame({"histos": harray2})

    # correctness of these functions is checked in test_spark_executor
    agg_histos_raw(series1, 1)
    agg_histos_raw(series2, 1)
    reduce_histos_raw(df, 1)
