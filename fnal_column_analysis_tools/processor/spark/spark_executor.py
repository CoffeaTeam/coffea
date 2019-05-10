from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor,as_completed

import tqdm
import cloudpickle

import pyspark
import pyspark.sql.functions as fn
from pyspark.sql.types import DoubleType,BinaryType

import lz4.frame as lz4f

from jinja2 import Environment, PackageLoader, select_autoescape

lz4_clevel = 1

#this is a UDF that takes care of summing histograms across
#various spark results where the outputs are histogram blobs
@fn.pandas_udf(BinaryType(), fn.PandasUDFType.GROUPED_AGG)
def agg_histos(df):
    global lz4_clevel
    goodlines = df[df.str.len() > 0]
    outhist = None
    for line in goodlines:
        temp = cpkl.loads(lz4f.decompress(line))
        if outhist is None:
            outhist = temp
        else:
            for key,val in temp.items():
                outhist[key] += val
    return lz4f.compress(cpkl.dumps(outhist),compression_level=lz4_clevel)


class SparkExecutor(object):
    _template_name = 'spark_template.py'
    
    def __init__(self):
        self._cacheddfs = None
        self._rawresults = None
        self._env = Environment(loader=PackageLoader('fnal_column_analysis_tools',
                                                     'processor',
                                                     'templates'),
                                autoescape=select_autoescape(['py'])
                                )

    def __call__(self, spark, dfslist, processor_instance, output, thread_workers,
                 unit='datasets', desc='Processing'):
        #get columns from processor
        columns = ['dataset'] + processor_instance.columns
        #cache the input datasets if it's not already done
        if self._cacheddfs is None:
            self._cacheddfs = {}
            for ds,df in dfslist.items():
                self._cacheddfs[ds] = df.select(*tuple(columns)).cache()

        with ThreadPoolExecutor(max_workers=thread_workers) as executor:
            future_to_ds = {}
            for ds,df in self.cacheddfs.items():
                future_to_ds[executor.submit(self._launch_analysis,df,columns)] = ds
            #wait for the spark jobs to come in
            self._rawresults = {}
            for future in tqdm(as_completed(future_to_ds),
                               total=len(future_to_ds),
                               desc=desc,unit=unit):
                self._rawresults[future_to_ds[future]] = future.result()
            
        results = deepcopy(list(self.rawresults.values()))
        start = results.pop()
        start = start[start.columns[0]][0]
        output = cpkl.loads(lz4f.decompress(start))
        for bitstream in myresults:
            if bitstream.empty: continue
            bits = bitstream[bitstream.columns[0]][0]
            for key,ahist in cpkl.loads(lz4f.decompress(bits)).items():
                output[key] += ahist
            

    def _launch_analysis(self, df, columns):
        return df.withColumn('partid', fn.spark_partition_id()) \
                 .withColumn('histos', compute(*tuple(columns))) \
                 .select('partid','histos') \
                 .groupBy('partid').agg(agg_histos('histos')) \
                 .groupBy().agg(agg_histos('agg_histos(histos)')) \
                 .toPandas()

    def _wrap_processor(self,processor_instance):
        pass

spark_executor = SparkExecutor()
