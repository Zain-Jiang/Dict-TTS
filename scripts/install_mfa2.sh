#!/bin/bash
set -e
pip uninstall typing
pip install --ignore-requires-python git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git@v2.0.0b3
mfa thirdparty download
conda intall -c conda-forge openblas openfst
sudo apt-get install libopenblas-base