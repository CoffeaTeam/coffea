#!/usr/bin/env bash

pip install -U sphinx nbsphinx sphinx-rtd-theme
python setup.py install
pushd docs
pushd source
sphinx-autogen reference.rst
popd
make html
touch build/html/.nojekyll
popd
pip uninstall --yes coffea 
