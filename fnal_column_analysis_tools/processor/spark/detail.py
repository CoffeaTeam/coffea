from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pyspark.sql
import pyspark.sql.functions as fn
from collections.abc import Sequence

# this is a reasonable local spark configuration
_default_config = pyspark.sql.SparkSession.builder \
    .appName('coffea-analysis') \
    .master('local[*]') \
    .config('spark.sql.execution.arrow.enabled', 'true') \
    .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)


def _spark_initialize(config=_default_config, **kwargs):
    spark_progress = False
    if 'spark_progress' in kwargs.keys():
        spark_progress = kwargs['spark_progress']

    cfg_actual = config
    if not spark_progress:
        cfg_actual = cfg_actual.config('spark.ui.showConsoleProgress', 'false')

    session = cfg_actual.getOrCreate()
    sc = session.sparkContext

    if 'log_level' in kwargs.keys():
        sc.setLogLevel(kwargs['log_level'])
    else:
        sc.setLogLevel('ERROR')

    return session


def _read_df(spark, files_or_dirs):
    if not isinstance(files_or_dirs, Sequence):
        raise ValueError("spark dataset file list must be a Sequence (like list())")
    return spark.read.parquet(*files_or_dirs)


def _spark_make_dfs(spark, fileset, partitionsize, thread_workers):
    dfs = {}
    with ThreadPoolExecutor(max_workers=thread_workers) as executor:
        future_to_ds = {executor.submit(_read_df, spark, fileset[dataset]): dataset for dataset in fileset.keys()}
        for ftr in tqdm(as_completed(future_to_ds), total=len(fileset), desc='loading', unit='datasets'):
            dataset = future_to_ds[ftr]
            df = ftr.result()
            df = df.withColumn('dataset', fn.lit(dataset))
            count = df.count()
            npartitions = max(count // partitionsize, 1)
            if df.rdd.getNumPartitions() > npartitions:
                df = df.coalesce(npartitions)
            dfs[dataset] = df
    return dfs


def _spark_stop(spark):
    # this may do more later?
    spark.stop()
