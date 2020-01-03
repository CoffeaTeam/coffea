#!/usr/bin/env bash

python -m pip -q install -U sphinx==2.2.2 nbsphinx sphinx-rtd-theme sphinx-automodapi
pip -q install -e .
pushd docs
rm -rf build
pushd source
rm -rf api
rm -rf modules
sphinx-autogen -t _templates reference.rst
popd
make html
touch build/html/.nojekyll
popd
