from coffea import hist, processor
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pickle as pkl
import lz4.frame as lz4f
import numpy as np
import pandas as pd
from functools import partial

from ..executor import _futures_handler
from coffea.nanoaod import NanoEvents

import pyspark
import pyspark.sql.functions as fn
from pyspark.sql.types import BinaryType, StringType, StructType, StructField

from jinja2 import Environment, PackageLoader, select_autoescape
from coffea.util import awkward

lz4_clevel = 1


# this is a UDF that takes care of summing histograms across
# various spark results where the outputs are histogram blobs
def agg_histos_raw(series, processor_instance, lz4_clevel):
    goodlines = series[series.str.len() > 0]
    if goodlines.size == 1:  # short-circuit trivial aggregations
        return goodlines[0]
    outhist = processor_instance.accumulator.identity()
    for line in goodlines:
        outhist.add(pkl.loads(lz4f.decompress(line)))
    return lz4f.compress(pkl.dumps(outhist), compression_level=lz4_clevel)


@fn.pandas_udf(BinaryType(), fn.PandasUDFType.GROUPED_AGG)
def agg_histos(series):
    global processor_instance, lz4_clevel
    return agg_histos_raw(series, processor_instance, lz4_clevel)


def reduce_histos_raw(df, processor_instance, lz4_clevel):
    histos = df['histos']
    mask = (histos.str.len() > 0)
    outhist = processor_instance.accumulator.identity()
    for line in histos[mask]:
        outhist.add(pkl.loads(lz4f.decompress(line)))
    return pd.DataFrame(data={'histos': np.array([lz4f.compress(pkl.dumps(outhist), compression_level=lz4_clevel)], dtype='O')})


@fn.pandas_udf(StructType([StructField('histos', BinaryType(), True)]), fn.PandasUDFType.GROUPED_MAP)
def reduce_histos(df):
    global processor_instance, lz4_clevel
    return reduce_histos_raw(df, processor_instance, lz4_clevel)


class SparkExecutor(object):
    _template_name = 'spark.py.tmpl'

    def __init__(self):
        self._cacheddfs = None
        self._rawresults = None
        self._counts = None
        self._env = Environment(loader=PackageLoader('coffea.processor',
                                                     'templates'),
                                autoescape=select_autoescape(['py'])
                                )

    @property
    def counts(self):
        return self._counts

    def __call__(self, spark, dfslist, theprocessor, output, thread_workers,
                 use_df_cache, flatten, nano, status=True, unit='datasets', desc='Processing'):
        # processor needs to be a global
        global processor_instance, coffea_udf, coffea_udf_flat, coffea_udf_nano
        processor_instance = theprocessor
        # get columns from processor
        columns = processor_instance.columns
        cols_w_ds = ['dataset'] + columns
        # make our udf
        tmpl = self._env.get_template(self._template_name)
        render = tmpl.render(cols=columns)
        exec(render)

        # cache the input datasets if it's not already done
        if self._counts is None:
            self._counts = {}
            # go through each dataset and thin down to the columns we want
            for ds, (df, counts) in dfslist.items():
                self._counts[ds] = counts

        def spex_accumulator(total, result):
            ds, df = result
            total[ds] = df

        if self._cacheddfs is None:
            self._cacheddfs = {}
            cachedesc = 'caching' if use_df_cache else 'pruning'
            with ThreadPoolExecutor(max_workers=thread_workers) as executor:
                futures = set()
                for ds, (df, counts) in dfslist.items():
                    futures.add(executor.submit(self._pruneandcache_data, ds, df, cols_w_ds, use_df_cache))
                _futures_handler(futures, self._cacheddfs, status, unit, cachedesc, spex_accumulator, None)

        with ThreadPoolExecutor(max_workers=thread_workers) as executor:
            futures = set()
            for ds, df in self._cacheddfs.items():
                co_udf = coffea_udf
                if flatten:
                    co_udf = coffea_udf_flat
                if nano:
                    co_udf = coffea_udf_nano
                futures.add(executor.submit(self._launch_analysis, ds, df, co_udf, cols_w_ds))
            # wait for the spark jobs to come in
            self._rawresults = {}
            _futures_handler(futures, self._rawresults, status, unit, desc, spex_accumulator, None)

        for ds, bitstream in self._rawresults.items():
            if bitstream is None:
                raise Exception('No pandas dataframe returned from spark in dataset: %s, something went wrong!' % ds)
            if bitstream.empty:
                raise Exception('The histogram list returned from spark is empty in dataset: %s, something went wrong!' % ds)
            bits = bitstream[bitstream.columns[0]][0]
            output.add(pkl.loads(lz4f.decompress(bits)))

    def _pruneandcache_data(self, ds, df, columns, cacheit):
        if cacheit:
            return ds, df.select(*columns).cache()
        return ds, df.select(*columns)

    def _launch_analysis(self, ds, df, udf, columns):
        histo_map_parts = (df.rdd.getNumPartitions() // 20) + 1
        return ds, df.select(udf(*columns).alias('histos')) \
                     .withColumn('hpid', fn.spark_partition_id() % histo_map_parts) \
                     .repartition(histo_map_parts, 'hpid') \
                     .groupBy('hpid').apply(reduce_histos) \
                     .groupBy().agg(agg_histos('histos')) \
                     .toPandas()


spark_executor = SparkExecutor()
