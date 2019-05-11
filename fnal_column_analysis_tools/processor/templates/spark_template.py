global coffea_udf

@fn.pandas_udf(BinaryType(), fn.PandasUDFType.SCALAR)
def coffea_udf({% for col in cols %}{{col}}{{ "," if not loop.last }}{% endfor %}):
    global processor_instance, lz4_clevel
    
    columns = [{% for col in cols %}{{col}}{{ "," if not loop.last }}{% endfor %}]
    names = [{% for col in cols %}{{"'"|safe+col+"'"|safe}}{{ "," if not loop.last }}{% endfor %}]
    
    for i,col in enumerate(columns):
        #numpy array
        if names[i] == 'dataset' or columns[i].array[0].base is None:
            columns[i] = columns[i].values
        else:
            columns[i] = columns[i].array[0].base

    size = dataset.values.size
    items = {name:col for name,col in zip(names,columns)}
    
    df = processor.PreloadedDataFrame(size=size, items=items)
    df['dataset'] = dataset[0]
    
    vals = processor_instance.process(df)
    
    valsblob = lz4f.compress(cpkl.dumps(vals),compression_level=lz4_clevel)
    
    outs = np.full(shape=(size,),fill_value=b'',dtype='O')
    outs[0] = valsblob
    
    return pd.Series(outs)
