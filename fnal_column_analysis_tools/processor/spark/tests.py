from fnal_column_analysis_tools.processor.spark.detail import (_spark_initialize,
                                                               _spark_stop)

import pyspark.sql.functions as fn
from pyspark.sql.types import DoubleType, BinaryType
import lz4.frame as lz4f
import cloudpickle as cpkl
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from fnal_column_analysis_tools import hist, processor
from fnal_column_analysis_tools.analysis_objects import JaggedCandidateArray as CandArray
import numpy as np


class DummyProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        mass_axis = hist.Bin("mass", r"$m_{\mu\mu}$ [GeV]", 30000, 0.25, 300)

        self._accumulator = processor.dict_accumulator({
                                                       'mass': hist.Hist("Counts", dataset_axis, mass_axis),
                                                       'cutflow': processor.defaultdict_accumulator(int),
                                                       })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()

        dataset = df['dataset']

        output['mass'].fill(dataset=dataset, mass=125.0)
        output['cutflow']['dummy'] += 1

        return output

    def postprocess(self, accumulator):
        return accumulator


def check_spark_functionality():
    spark = _spark_initialize()

    env = Environment(loader=PackageLoader('fnal_column_analysis_tools.processor',
                                           'templates'),
                      autoescape=select_autoescape(['py'])
                      )

    template_name = 'spark_template.py'
    tmpl = env.get_template(template_name)

    global processor_instance, lz4_clevel, coffea_udf
    processor_instance = DummyProcessor()
    lz4_clevel = 1

    cols = ['dataset']
    output = tmpl.render(cols=cols)
    exec(output)

    dataset = [{'dataset': 'WJets'}, {'dataset': 'WJets'}, {'dataset': 'WJets'}]
    df = spark.createDataFrame(dataset, schema='dataset: string')
    pd_one = df.toPandas()

    df = df.withColumn('histos', coffea_udf(*cols))
    pd_two = df.toPandas()

    _spark_stop(spark)

    return pd_one['dataset'].count(), pd_two['dataset'].count(), pd_two['histos']


def run_spark_executor():
    pass


class NanoTestProcessor(processor.ProcessorABC):
    def __init__(self, columns):
        self._columns = ['dataset'] + columns
        dataset_axis = hist.Cat("dataset", "Primary dataset")
        mass_axis = hist.Bin("mass", r"$m_{\mu\mu}$ [GeV]", 30000, 0.25, 300)
        pt_axis = hist.Bin("pt", r"$p_{T}$ [GeV]", 30000, 0.25, 300)

        self._accumulator = processor.dict_accumulator({
                                                       'mass': hist.Hist("Counts", dataset_axis, mass_axis),
                                                       'pt': hist.Hist("Counts", dataset_axis, pt_axis),
                                                       'cutflow': processor.defaultdict_accumulator(int),
                                                       })

    @property
    def columns(self):
        return self._columns

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()

        dataset = df['dataset']

        muon = CandArray.candidatesfromcounts(counts=df['nMuon'],
                                              pt=df['Muon_pt'],
                                              eta=df['Muon_eta'],
                                              phi=df['Muon_phi'],
                                              mass=df['Muon_mass'])
        dimuon = muon.distincts()

        output['pt'].fill(dataset=dataset, pt=muon.pt.flatten())
        output['mass'].fill(dataset=dataset, mass=dimuon.mass.flatten())
        output['cutflow']['%s_pt' % dataset] += np.sum(muon.counts)
        output['cutflow']['%s_mass' % dataset] += np.sum(dimuon.counts)

        return output

    def postprocess(self, accumulator):
        return accumulator
