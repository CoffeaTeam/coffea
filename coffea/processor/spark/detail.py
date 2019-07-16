from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pyspark.sql
import pyspark.sql.functions as fn
from pyarrow.compat import guid
from collections.abc import Sequence

from ..executor import futures_handler

# this is a reasonable local spark configuration
_default_config = pyspark.sql.SparkSession.builder \
    .appName('coffea-analysis-%s' % guid()) \
    .master('local[*]') \
    .config('spark.sql.execution.arrow.enabled', 'true') \
    .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)


def _spark_initialize(config=_default_config, **kwargs):
    spark_progress = False
    if 'spark_progress' in kwargs.keys():
        spark_progress = kwargs['spark_progress']

    cfg_actual = config
    # get spark to not complain about missing log configs
    cfg_actual = cfg_actual.config('spark.driver.extraJavaOptions', '-Dlog4jspark.root.logger=ERROR,console')
    if not spark_progress:
        cfg_actual = cfg_actual.config('spark.ui.showConsoleProgress', 'false')

    # always load laurelin even if we may not use it
    kwargs.setdefault('laurelin_version', '0.1.0')
    laurelin = kwargs['laurelin_version']
    cfg_actual = cfg_actual.config('spark.jars.packages',
                                   'edu.vanderbilt.accre:laurelin:%s' % laurelin)

    session = cfg_actual.getOrCreate()
    sc = session.sparkContext

    if 'log_level' in kwargs.keys():
        sc.setLogLevel(kwargs['log_level'])
    else:
        sc.setLogLevel('ERROR')

    return session


def _read_df(spark, dataset, files_or_dirs, ana_cols, partitionsize, file_type, treeName='Events'):
    if not isinstance(files_or_dirs, Sequence):
        raise ValueError('spark dataset file list must be a Sequence (like list())')
    df = None
    if file_type == 'parquet':
        df = spark.read.parquet(*files_or_dirs)
    else:
        df = spark.read.format(file_type) \
                       .option('tree', treeName) \
                       .load(*files_or_dirs)
    count = df.count()

    df_cols = set(df.columns)
    cols_in_df = ana_cols.intersection(df_cols)
    df = df.select(*cols_in_df)
    missing_cols = ana_cols - cols_in_df
    for missing in missing_cols:
        df = df.withColumn(missing, fn.lit(0.0))
    df = df.withColumn('dataset', fn.lit(dataset))
    npartitions = (count // partitionsize) + 1
    if df.rdd.getNumPartitions() > npartitions:
        df = df.repartition(npartitions)

    return df, dataset, count


def _spark_make_dfs(spark, fileset, partitionsize, columns, thread_workers, file_type, status=True):
    dfs = {}
    ana_cols = set(columns)

    def dfs_accumulator(total, result):
        df, ds, count = result
        total[ds] = (df, count)

    with ThreadPoolExecutor(max_workers=thread_workers) as executor:
        futures = set(executor.submit(_read_df, spark, ds, files,
                                      ana_cols, partitionsize, file_type) for ds, files in fileset.items())

        futures_handler(futures, dfs, status, 'datasets', 'loading', futures_accumulator=dfs_accumulator)

    return dfs


def _spark_stop(spark):
    # this may do more later?
    spark._jvm.SparkSession.clearActiveSession()
    spark.stop()
