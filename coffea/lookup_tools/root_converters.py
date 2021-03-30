from ..util import numpy as np
import uproot
import re

cycle = re.compile(r";\d+")


def killcycle(s, cycle):
    return cycle.sub("", s)


histTypes = ["TH1D", "TH1F", "TH2D", "TH2F", "TH3D", "TH3F"]
graphTypes = ["TGraphAsymmErrors", "TGraph2D"]


def convert_histo_root_file(file):
    converted_file = {}
    fin = uproot.open(file.strip())
    for path, item in fin.iteritems(recursive=True):
        nicepath = killcycle(path, cycle)
        rootclass = item.classname
        if rootclass in histTypes:
            edges = tuple(ax.edges() for ax in item.axes)
            converted_file[(nicepath, "dense_lookup")] = item.values(), edges
            if hasattr(item, "variances"):
                converted_file[(nicepath + "_error", "dense_lookup")] = (
                    np.sqrt(item.variances()),
                    edges,
                )
        elif rootclass in graphTypes:
            # TODO: convert TGraph into interpolated lookup
            continue
        elif hasattr(item, "_fEX") and hasattr(item, "_fEY"):
            # TODO what is this?
            tempArrX = item._fEX
            tempArrY = item._fEY
            converted_file[(nicepath, "dense_lookup")] = [tempArrX, tempArrY]

    return converted_file
