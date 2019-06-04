from fnal_column_analysis_tools.util import awkward
from fnal_column_analysis_tools.util import numpy as np

# pt except for reshaping, then discriminant
btag_feval_dims = {0: [1], 1: [1], 2: [1], 3: [2]}


def convert_btag_csv_file(csvFilePath):
    btag_f = open(csvFilePath)
    nameandcols = btag_f.readline().split(';')
    btag_f.close()
    name = nameandcols[0].strip()
    columns = nameandcols[1].strip()
    columns = [column.strip() for column in columns.split(',')]

    corrections = np.genfromtxt(csvFilePath,
                                dtype=None,
                                names=tuple(columns),
                                converters={1: lambda s: s.strip(),
                                            2: lambda s: s.strip(),
                                            10: lambda s: s.strip(' "')},
                                delimiter=',',
                                skip_header=1,
                                unpack=True,
                                encoding='ascii'
                                )

    all_names = corrections[[columns[i] for i in range(4)]]
    labels = np.unique(corrections[[columns[i] for i in range(4)]])
    wrapped_up = {}
    for label in labels:
        etaMins = np.unique(corrections[np.where(all_names == label)][columns[4]])
        etaMaxs = np.unique(corrections[np.where(all_names == label)][columns[5]])
        etaBins = np.union1d(etaMins, etaMaxs)
        ptMins = np.unique(corrections[np.where(all_names == label)][columns[6]])
        ptMaxs = np.unique(corrections[np.where(all_names == label)][columns[7]])
        ptBins = np.union1d(ptMins, ptMaxs)
        discrMins = np.unique(corrections[np.where(all_names == label)][columns[8]])
        discrMaxs = np.unique(corrections[np.where(all_names == label)][columns[9]])
        discrBins = np.union1d(discrMins, discrMaxs)
        vals = np.zeros(shape=(len(discrBins) - 1, len(ptBins) - 1, len(etaBins) - 1),
                        dtype=corrections.dtype[10])
        for i, eta_bin in enumerate(etaBins[:-1]):
            for j, pt_bin in enumerate(ptBins[:-1]):
                for k, discr_bin in enumerate(discrBins[:-1]):
                    this_bin = np.where((all_names == label) &
                                        (corrections[columns[4]] == eta_bin) &
                                        (corrections[columns[6]] == pt_bin) &
                                        (corrections[columns[8]] == discr_bin))
                    vals[k, j, i] = corrections[this_bin][columns[10]][0]
        label_decode = []
        for i in range(len(label)):
            label_decode.append(label[i])
            if isinstance(label_decode[i], bytes):
                label_decode[i] = label_decode[i].decode()
            else:
                label_decode[i] = str(label_decode[i])
        str_label = '_'.join([name] + label_decode)
        feval_dim = btag_feval_dims[label[0]]
        wrapped_up[(str_label, 'dense_evaluated_lookup')] = (vals, (etaBins, ptBins, discrBins), tuple(feval_dim))
    return wrapped_up
