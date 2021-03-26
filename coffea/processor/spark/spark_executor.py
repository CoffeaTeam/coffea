from coffea import hist, processor
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
import pickle
import lz4.frame
import numpy
import pandas
import awkward
from functools import partial

from coffea.processor.executor import _futures_handler, _decompress, _reduce
from coffea.processor.accumulator import accumulate
from coffea.nanoevents import NanoEventsFactory, schemas
from coffea.nanoevents.mapping import SimplePreloadedColumnSource

import pyspark
import pyspark.sql.functions as fn
from pyspark.sql.types import BinaryType, StringType, StructType, StructField

from jinja2 import Environment, PackageLoader, select_autoescape
from coffea.util import awkward

lz4_clevel = 1


# this is a UDF that takes care of summing histograms across
# various spark results where the outputs are histogram blobs
def agg_histos_raw(series, lz4_clevel):
    goodlines = series[series.str.len() > 0]
    if goodlines.size == 1:  # short-circuit trivial aggregations
        return goodlines[0]
    return _reduce(lz4_clevel)(goodlines)


@fn.pandas_udf(BinaryType(), fn.PandasUDFType.GROUPED_AGG)
def agg_histos(series):
    global lz4_clevel
    return agg_histos_raw(series, lz4_clevel)


def reduce_histos_raw(df, lz4_clevel):
    histos = df['histos']
    outhist = _reduce(lz4_clevel)(histos[histos.str.len() > 0])
    return pandas.DataFrame(data={'histos': numpy.array([outhist], dtype='O')})


@fn.pandas_udf(StructType([StructField('histos', BinaryType(), True)]), fn.PandasUDFType.GROUPED_MAP)
def reduce_histos(df):
    global lz4_clevel
    return reduce_histos_raw(df, lz4_clevel)


def _get_ds_bistream(item):
    global lz4_clevel
    ds, bitstream = item
    if bitstream is None:
        raise Exception('No pandas dataframe returned from spark in dataset: %s, something went wrong!' % ds)
    if bitstream.empty:
        raise Exception('The histogram list returned from spark is empty in dataset: %s, something went wrong!' % ds)
    out = bitstream[bitstream.columns[0]][0]
    if lz4_clevel is not None:
        return _decompress(out)
    return out


class SparkExecutor(object):
    _template_name = 'spark.py.tmpl'

    def __init__(self):
        self._cacheddfs = None
        self._counts = None
        self._env = Environment(loader=PackageLoader('coffea.processor',
                                                     'templates'),
                                autoescape=select_autoescape(['py'])
                                )

    @property
    def counts(self):
        return self._counts

    def __call__(self, spark, dfslist, theprocessor, output, thread_workers,
                 use_df_cache, schema, status=True, unit='datasets', desc='Processing'):
        # processor needs to be a global
        global processor_instance, coffea_udf, nano_schema
        processor_instance = theprocessor
        if schema is None:
            schema = schemas.BaseSchema
        if not issubclass(schema, schemas.BaseSchema):
            raise ValueError("Expected schema to derive from BaseSchema (%s)" % (str(schema.__name__)))
        nano_schema = schema
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

        if self._cacheddfs is None:
            self._cacheddfs = {}
            cachedesc = 'caching' if use_df_cache else 'pruning'
            with ThreadPoolExecutor(max_workers=thread_workers) as executor:
                futures = set()
                for ds, (df, counts) in dfslist.items():
                    futures.add(executor.submit(self._pruneandcache_data, ds, df, cols_w_ds, use_df_cache))
                gen = _futures_handler(futures, timeout=None)
                try:
                    for ds, df in tqdm(
                        gen,
                        disable=not status,
                        unit=unit,
                        total=len(dfslist),
                        desc=cachedesc,
                    ):
                        self._cacheddfs[ds] = df
                finally:
                    gen.close()

        with ThreadPoolExecutor(max_workers=thread_workers) as executor:
            futures = set()
            for ds, df in self._cacheddfs.items():
                co_udf = coffea_udf
                futures.add(executor.submit(self._launch_analysis, ds, df, co_udf, cols_w_ds))
            gen = _futures_handler(futures, timeout=None)
            try:
                output = accumulate(
                    tqdm(
                        map(_get_ds_bistream, gen),
                        disable=not status,
                        unit=unit,
                        total=len(self._cacheddfs),
                        desc=desc,
                    ),
                    output,
                )
            finally:
                gen.close()

        return output

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
