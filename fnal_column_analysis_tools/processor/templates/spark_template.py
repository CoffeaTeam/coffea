global coffea_udf

@fn.pandas_udf(BinaryType(), fn.PandasUDFType.SCALAR)
def coffea_udf({% for col in cols %}{{col}}{{ "," if not loop.last }}{% endfor %}):
    global processor_instance, lz4_clevel
        
    columns = [{% for col in cols %}{{col}}{{ "," if not loop.last }}{% endfor %}]
    names = [{% for col in cols %}{{"'"|safe+col+"'"|safe}}{{ "," if not loop.last }}{% endfor %}]
    
    size = dataset.values.size
    items = {name:col.values for name,col in zip(names,columns)}
    
    df = processor.PreloadedDataFrame(size=size, items=items)
    df['dataset'] = dataset[0]
    
    vals = processor_instance.process(df)
    
    valsblob = lz4f.compress(cpkl.dumps(vals),compression_level=lz4_clevel)
    
    outs = np.full(shape=(size,),fill_value=b'',dtype='O')
    outs[0] = valsblob
    
    return pd.Series(outs)
