import numpy as np
import uproot

TH1D = "<class 'uproot.rootio.TH1D'>"
TH2D = "<class 'uproot.rootio.TH2D'>"
TH1F = "<class 'uproot.rootio.TH1F'>"
TH2F = "<class 'uproot.rootio.TH2F'>"
TGraphAsymmErrors = "<class 'uproot.rootio.TGraphAsymmErrors'>"
RootDir = "<class 'uproot.rootio.ROOTDirectory'>"

def convert_histo_root_file(file):
    converted_file = {}
    dumFile = uproot.open(file.strip())
    for i, key in enumerate(dumFile.keys()):
        histType = str(type(dumFile[key[:-2]]))
        if histType == TH1D or histType == TH2D or histType==TH1F or histType==TH2F:
            if  not ("bound method" in str(dumFile[key[:-2]].numpy)):
                converted_file[(key[:-2],'dense_lookup')] = dumFile[key[:-2]].numpy
            else:
                converted_file[(key[:-2],'dense_lookup')] = dumFile[key[:-2]].numpy()
        elif histType == RootDir: #means there are subdirectories wihin main directory
            for j, key2 in enumerate(dumFile[key[:-2]].keys()):
                histType2 = str(type(dumFile[key[:-2]][key2[:-2]]))
                if histType2 == TH1D or histType2 == TH2D or histType2==TH1F or histTyp2==TH2F:
                    if not ("bound method" in str(dumFile[key[:-2]][key2[:-2]].numpy)):
                        converted_file[(key[:-2]+'/'+key2[:-2],'dense_lookup')] = dumFile[key[:-2]][key2[:-2]].numpy
                    else:
                        converted_file[(key[:-2]+'/'+key2[:-2],'dense_lookup')] = dumFile[key[:-2]][key2[:-2]].numpy()
        elif histType==TGraphAsymmErrors:
            continue
        else:
            tempArrX= dumFile[key[:-2]]._fEX
            tempArrY= dumFile[key[:-2]]._fEY
            converted_file[(key[:-2],'dense_lookup')] = [tempArrX, tempArrY]
    return converted_file

