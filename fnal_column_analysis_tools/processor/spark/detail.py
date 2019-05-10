from concurrent.futures import ThreadPoolExecutor,as_completed

import pyspark.sql
import pyspark.sql.functions as fn
from pyspark.sql.types import DoubleType, BinaryType

#this is a reasonable local spark configuration
_default_config = pyspark.sql.SparkSession.builder \
    .appName('coffea-analysis') \
    .master('local[*]') \
    .config('spark.sql.execution.arrow.enabled','true') \
    .config('spark.sql.execution.arrow.maxRecordsPerBatch', 200000)

def _spark_initialize(config=_default_config,**kwargs):
    session = config.getOrCreate()
    sc = session.sparkContext

    if 'log_level' in kwargs.keys():
        sc.getLogLevel(kwargs['log_level'])
    
    return session

def _spark_make_dfs(spark, filelist, columns, thread_workers, file_list=False):
    with ThreadPoolExecutor(max_workers=thread_workers) as executor:
        future_to_ds = {executor.submit(read_df,dataset): dataset for dataset in datasets.keys()}
        for ftr in as_completed(future_to_ds):
            dataset = future_to_ds[future]
            df = future.result()
            df = df.withColumn('dataset', fn.lit(dataset))
    
    return spark.read.parquet('hdfs:///store/parquet/zprimebits/%s/%s/'%(skim_root,dsloc))

def _spark_stop(spark):
    #this may do more later?
    spark.stop()
