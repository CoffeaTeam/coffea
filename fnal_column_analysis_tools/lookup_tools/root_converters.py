import numpy as np
import uproot
import re

cycle = re.compile(br";\d+")
killcycle = lambda s: cycle.sub(b"", s)

histTypes = [b'TH1D', b'TH1F', b'TH2D', b'TH2F']
graphTypes = [b'TGraphAsymmErrors', b'TGraph2D']


def convert_histo_root_file(file):
    converted_file = {}
    fin = uproot.open(file.strip())
    for path, item in fin.iteritems(recursive=True):
        nicepath = killcycle(path)
        rootclass = item._classname
        if rootclass in histTypes:
            converted_file[(nicepath, 'dense_lookup')] = item.values, item.edges
        elif rootclass in graphTypes:
            # TODO: convert TGraph into interpolated lookup
            continue
        elif hasattr(item, '_fEX') and hasattr(item, '_fEY'):
            # TODO what is this?
            tempArrX = item._fEX
            tempArrY = item._fEY
            converted_file[(nicepath, 'dense_lookup')] = [tempArrX, tempArrY]

    return converted_file

