import pytest

from coffea import processor


def do_dask_pandas_job(client, filelist):
    from coffea import nanoevents
    from coffea.processor.test_items import NanoTestProcessorPandas

    executor = processor.DaskExecutor(client=client, use_dataframes=True)
    run = processor.Runner(executor=executor, schema=nanoevents.NanoAODSchema)

    output = run(filelist, "Events", processor_instance=NanoTestProcessorPandas())

    # Can save to Parquet straight from distributed DataFrame without explicitly collecting the outputs:
    #
    # import dask.dataframe as dd
    # dd.to_parquet(df=output, path=/output/path/)
    #
    #
    # It's also possible to do some operations on distributed DataFrames without collecting them.
    # For example, split the dataframe by column value and save to different directories:
    #
    # dd.to_parquet(df=output[output.dataset=='ZJets'], path=/output/path/ZJets/)
    # dd.to_parquet(df=output[output.dataset=='Data'], path=/output/path/Data/)

    # Alternatively, can continue working with output.
    # Convert from Dask DataFrame back to Pandas:
    output = output.compute()

    assert output[output.dataset == "ZJets"].shape[0] == 6
    assert output[output.dataset == "Data"].shape[0] == 18

    # print(output)


@pytest.mark.skip(reason="pandas dataframe output very different with dask-awkward")
def test_dask_pandas_job():
    distributed = pytest.importorskip("distributed", minversion="2.6.0")
    client = distributed.Client(dashboard_address=None)

    import os
    import os.path as osp

    filelist = {
        "ZJets": [osp.join(os.getcwd(), "tests/samples/nano_dy.root")],
        "Data": [osp.join(os.getcwd(), "tests/samples/nano_dimuon.root")],
    }

    do_dask_pandas_job(client, filelist)

    client.close()
