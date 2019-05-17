from fnal_column_analysis_tools import hist, processor
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import cloudpickle as cpkl
import lz4.frame as lz4f
import numpy as np
import pandas as pd

import pyspark
import pyspark.sql.functions as fn
from pyspark.sql.types import BinaryType, StructType, StructField

from jinja2 import Environment, PackageLoader, select_autoescape

lz4_clevel = 1

# this is a UDF that takes care of summing histograms across
# various spark results where the outputs are histogram blobs
@fn.pandas_udf(BinaryType(), fn.PandasUDFType.GROUPED_AGG)
def agg_histos(df):
    global processor_instance, lz4_clevel
    goodlines = df[df.str.len() > 0]
    outhist = processor_instance.accumulator.identity()
    for line in goodlines:
        outhist.add(cpkl.loads(lz4f.decompress(line)))
    return lz4f.compress(cpkl.dumps(outhist), compression_level=lz4_clevel)


@fn.pandas_udf(StructType([StructField('histos', BinaryType(), True)]), fn.PandasUDFType.GROUPED_MAP)
def remove_zeros(df):
    histos = df['histos']
    return pd.DataFrame(data=histos[histos.str.len() > 0], columns=['histos'])


class SparkExecutor(object):
    _template_name = 'spark.py.tmpl'

    def __init__(self):
        self._cacheddfs = None
        self._rawresults = None
        self._counts = None
        self._env = Environment(loader=PackageLoader('fnal_column_analysis_tools.processor',
                                                     'templates'),
                                autoescape=select_autoescape(['py'])
                                )

    @property
    def counts(self):
        return self._counts

    def __call__(self, spark, dfslist, theprocessor, output, thread_workers,
                 unit='datasets', desc='Processing'):
        # processor needs to be a global
        global processor_instance, coffea_udf
        processor_instance = theprocessor
        # get columns from processor
        columns = processor_instance.columns
        cols_w_ds = ['dataset'] + columns
        # make our udf
        tmpl = self._env.get_template(self._template_name)
        render = tmpl.render(cols=columns)
        exec(render)

        # cache the input datasets if it's not already done
        if self._cacheddfs is None:
            self._cacheddfs = {}
            self._counts = {}
            # go through each dataset and thin down to the columns we want
            for ds, (df, counts) in dfslist.items():
                self._cacheddfs[ds] = df.select(*cols_w_ds).cache()
                self._counts[ds] = counts

        with ThreadPoolExecutor(max_workers=thread_workers) as executor:
            future_to_ds = {}
            for ds, df in self._cacheddfs.items():
                future_to_ds[executor.submit(self._launch_analysis, df, coffea_udf, cols_w_ds)] = ds
            # wait for the spark jobs to come in
            self._rawresults = {}
            for future in tqdm(as_completed(future_to_ds),
                               total=len(future_to_ds),
                               desc=desc, unit=unit):
                self._rawresults[future_to_ds[future]] = future.result()

        for ds, bitstream in self._rawresults.items():
            if bitstream is None:
                raise Exception('No pandas dataframe returns from spark in dataset: %s, something went wrong!' % ds)
            if bitstream.empty:
                raise Exception('The histogram list returned from spark is empty in dataset: %s, something went wrong!' % ds)
            bits = bitstream[bitstream.columns[0]][0]
            output.add(cpkl.loads(lz4f.decompress(bits)))

    def _launch_analysis(self, df, udf, columns):
        return df.select(udf(*columns).alias('histos')) \
                 .groupBy().apply(remove_zeros) \
                 .groupBy().agg(agg_histos('histos')) \
                 .toPandas()


spark_executor = SparkExecutor()
