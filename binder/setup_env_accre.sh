#!/usr/bin/env bash

python3.6 -m virtualenv py36
source py36/bin/activate

pip install --upgrade pip
pip install cloudpickle jinja2 llvmlite numba numpy pandas ipykernel matplotlib lz4 pyarrow tqdm py4j

pip install uproot
pip install --upgrade uproot-methods
# 1.14 is kindof old but pinned by other packages it seems
pip install --upgrade numpy scipy

# get dependencies for it
pip install fnal-column-analysis-tools[parsl,spark] --upgrade
# get latest and greatest
pip install https://github.com/Parsl/parsl/zipball/master
git clone -b topic_hats_scaleout https://github.com/CoffeaTeam/fnal-column-analysis-tools.git

# progressbar, sliders, etc.
jupyter nbextension enable --py widgetsnbextension

# issue with python3 bindings, see https://sft.its.cern.ch/jira/browse/SPI-1198
wget https://raw.githubusercontent.com/lgray/coffeandbacon/master/xrd_setup.py
wget https://github.com/xrootd/xrootd/archive/v4.8.5.tar.gz
tar zxf v4.8.5.tar.gz && rm -f v4.8.5.tar.gz
cp xrd_setup.py xrootd-4.8.5/bindings/python/
pushd xrootd-4.8.5/bindings/python/
python xrd_setup.py install
popd
rm -rf xrootd-4.8.5

ipython kernel install --user --name "py36_test" --display-name "py36_test"

deactivate

echo "environment \"py36\" installed, to start it do \"source `pwd`/bin/py36/activate\""
