from fnal_column_analysis_tools import hist, processor
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import _pickle as pkl
import lz4.frame as lz4f
import numpy as np
import pandas as pd

import pyspark
import pyspark.sql.functions as fn
from pyspark.sql.types import BinaryType, StringType, StructType, StructField

from jinja2 import Environment, PackageLoader, select_autoescape

lz4_clevel = 1

# this is a UDF that takes care of summing histograms across
# various spark results where the outputs are histogram blobs
# 用于负责汇总直方图的用户自定义方法,结果输出的是直方图中的点.

# 筛选出df中长度大于0的元素,并且对元素进行lz4_clevel等级的压缩.
@fn.pandas_udf(BinaryType(), fn.PandasUDFType.GROUPED_AGG)
def agg_histos(df):
    global processor_instance, lz4_clevel
    goodlines = df[df.str.len() > 0]
    # goodlines列表中只有一个元素时
    if goodlines.size == 1:  # short-circuit trivial aggregations
        return goodlines[0]
    outhist = processor_instance.accumulator.identity()
    # 循环取出列表中的元素添加到outhist中.
    for line in goodlines:
        outhist.add(pkl.loads(lz4f.decompress(line)))
    return lz4f.compress(pkl.dumps(outhist), compression_level=lz4_clevel)

# 筛选出histos中长度大于0的元素并进行压缩.
@fn.pandas_udf(StructType([StructField('histos', BinaryType(), True)]),
               fn.PandasUDFType.GROUPED_MAP)
def reduce_histos(df):
    histos = df['histos']
    mask = (histos.str.len() > 0)
    outhist = processor_instance.accumulator.identity()
    for line in histos[mask]:
        outhist.add(pkl.loads(lz4f.decompress(line)))
    return pd.DataFrame(data={'histos': np.array([lz4f.compress(pkl.dumps(outhist))], dtype='O')})


class SparkExecutor(object):
    _template_name = 'spark.py.tmpl'

    # 初始化方法
    def __init__(self):
        self._cacheddfs = None
        self._rawresults = None
        self._counts = None
        self._env = Environment(loader=PackageLoader('fnal_column_analysis_tools.processor',
                                                     'templates'),
                                autoescape=select_autoescape(['py'])
                                )
        
    # Python内置的@property装饰器就是负责把一个方法变成属性调用,这里将counts设置为了一个只可以读的属性.
    @property
    def counts(self):
        return self._counts

    def __call__(self, spark, dfslist, theprocessor, output, thread_workers,
                 unit='datasets', desc='Processing'):
        # processor needs to be a global 处理器需要的全局变量
        global processor_instance, coffea_udf
        processor_instance = theprocessor
        # get columns from processor 从处理器获得每列数据
        columns = processor_instance.columns
        cols_w_ds = ['dataset'] + columns
        # make our udf 
        tmpl = self._env.get_template(self._template_name)
        render = tmpl.render(cols=columns)
        exec(render)

        # cache the input datasets if it's not already done
        # 缓存输入数据集（如果尚未完成）
        if self._cacheddfs is None:
            self._cacheddfs = {}
            self._counts = {}
            # go through each dataset and thin down to the columns we want
            # 遍历每个数据集并向下细化到我们想要的列,其中对上面传入的dfslist字典类型数据调用items()函数,以列表返回可遍历的(键, 值).
            # eg: dict = {'Google': 'www.google.com', 'Runoob': 'www.runoob.com', 'taobao': 'www.taobao.com'}
            # dict.items()
            # [('Google', 'www.google.com'), ('taobao', 'www.taobao.com'), ('Runoob', 'www.runoob.com')]
            # ds是循环变量,df是key,counts是value.此for循环就是将字典dfslist的key和value进行分离的.
            for ds, (df, counts) in dfslist.items():
                self._cacheddfs[ds] = df.select(*cols_w_ds).cache()
                self._counts[ds] = counts
                
        # 此处为Python中的多线程,即开启thread_workers个线程同时工作.
        with ThreadPoolExecutor(max_workers=thread_workers) as executor:
            # 声明初始化一个字典类型变量
            future_to_ds = {}
            # 对上面的字典类型变量self._cacheddfs进行遍历取出元素.
            for ds, df in self._cacheddfs.items():
                future_to_ds[executor.submit(self._launch_analysis, df, coffea_udf, cols_w_ds)] = ds
            # wait for the spark jobs to come in
            # 等待spark作业进来
            # 声明初始化一个字典类型变量
            self._rawresults = {}
            # tqdm是python中很常用的模块，它的作用就是在终端上出现一个进度条，使得代码进度可视化。
            for future in tqdm(as_completed(future_to_ds),
                               total=len(future_to_ds),
                               desc=desc, unit=unit):
                self._rawresults[future_to_ds[future]] = future.result()
        
        # 通过for循环对字典中的脏数据(不存在和为空值的情况)进行清洗.
        for ds, bitstream in self._rawresults.items():
            if bitstream is None:
                raise Exception('No pandas dataframe returns from spark in dataset: %s, something went wrong!' % ds)
            if bitstream.empty:
                raise Exception('The histogram list returned from spark is empty in dataset: %s, something went wrong!' % ds)
            bits = bitstream[bitstream.columns[0]][0]
            output.add(pkl.loads(lz4f.decompress(bits)))
    # 没看懂??
    def _launch_analysis(self, df, udf, columns):
        histo_map_parts = (df.rdd.getNumPartitions() // 20) + 1
        return df.select(udf(*columns).alias('histos')) \
                 .withColumn('hpid', fn.spark_partition_id() % histo_map_parts) \
                 .repartition(histo_map_parts, 'hpid') \
                 .groupBy('hpid').apply(reduce_histos) \
                 .groupBy().agg(agg_histos('histos')) \
                 .toPandas()


spark_executor = SparkExecutor()
