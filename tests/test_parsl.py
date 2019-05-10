from __future__ import print_function, division
from fnal_column_analysis_tools import processor

import warnings

import numpy as np

def test_parsl_executor():
    try:
        import parsl
    except ModuleNotFoundError:
        warnings.warn('parsl not installed, skipping tests')
        return
    except Exception as e:
        warnings.warn('other error when trying to import parsl!')
        raise e
    
    from fnal_column_analysis_tools.processor import run_parsl_job
