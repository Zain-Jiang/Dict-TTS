#!/bin/bash
set -e

pip uninstall typing
if [ ! -f montreal-forced-aligner_linux.tar.gz ]; then
wget -c https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/releases/download/v1.0.1/montreal-forced-aligner_linux.tar.gz
fi
if [ ! -f v1.0.1.tar.gz ]; then
wget https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner/archive/v1.0.1.tar.gz
fi
tar xvf montreal-forced-aligner_linux.tar.gz
tar xvf v1.0.1.tar.gz
rm -rf mfa
mv Montreal-Forced-Aligner-1.0.1 mfa
cp scripts/mfa_aligner_textgrid.py mfa/aligner/textgrid.py # FIX: 跳过报错的句子，原版本报错会直接终止
export LD_LIBRARY_PATH=./montreal-forced-aligner/lib/:./montreal-forced-aligner/lib/thirdparty/bin/:$LD_LIBRARY_PATH
echo "| Install requirements."
cd mfa
pip install -r requirements.txt
# check if current user is superuser. not use sudo if run as superuser
if [[ $(id -u) -ne 0 ]]; then SUDO="sudo"; else SUDO=""; fi
$SUDO apt-get -y install libatlas3-base libgfortran3
echo "| freeze."
bash freezing/freeze.sh
cp -r ../montreal-forced-aligner/lib/thirdparty dist/montreal-forced-aligner/lib/
cd ../
