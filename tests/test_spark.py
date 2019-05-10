from fnal_column_analysis_tools import (hist,processor)

import warnings

import numpy as np

def test_spark_imports():
    try:
        import pyspark
    except ModuleNotFoundError:
        warnings.warn('pyspark not installed, skipping tests')
        return
    except Exception as e:
        warnings.warn('other error when trying to import pyspark!')
        raise e

    from fnal_column_analysis_tools.processor.spark.spark_executor import spark_executor
    from fnal_column_analysis_tools.processor.spark.detail import (_spark_initialize,
                                                                   _spark_make_dfs,
                                                                   _spark_stop)

    spark = _spark_initialize()
    _spark_stop(spark)

def test_spark_functionality():
    try:
        import pyspark
    except ModuleNotFoundError:
        warnings.warn('pyspark not installed, skipping tests')
        return
    except Exception as e:
        warnings.warn('other error when trying to import pyspark!')
        raise e
    
    from fnal_column_analysis_tools.processor.spark.tests import check_spark_functionality
    one,two = check_spark_functionality()

    assert(one == two)


def test_spark_executor():
    try:
        import pyspark
    except ModuleNotFoundError:
        warnings.warn('pyspark not installed, skipping tests')
        return
    except Exception as e:
        warnings.warn('other error when trying to import pyspark!')
        raise e

    from fnal_column_analysis_tools.processor import run_spark_job
