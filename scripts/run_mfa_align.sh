#!/bin/bash
set -e

NUM_JOB=${NUM_JOB:-36}
MFA_VERSION=${MFA_VERSION:-"1"}
echo "| Training MFA using ${NUM_JOB} cores."
BASE_DIR=data/processed/$CORPUS

mfa align $BASE_DIR/mfa_inputs $BASE_DIR/mfa_dict.txt $BASE_DIR/mfa_model.zip $BASE_DIR/mfa_outputs_tmp -t $BASE_DIR/mfa_tmp --clean -j $NUM_JOB --config_path data_gen/tts/mfa_config.yaml

rm -rf $BASE_DIR/mfa_tmp $BASE_DIR/mfa_outputs
mkdir -p $BASE_DIR/mfa_outputs
find $BASE_DIR/mfa_outputs_tmp -maxdepth 1 -regex ".*/[0-9]+" -print0 | xargs -0 -i rsync -a {}/ $BASE_DIR/mfa_outputs/
if [ -e "$BASE_DIR/mfa_outputs_tmp/unaligned.txt" ]; then
  cp $BASE_DIR/mfa_outputs_tmp/unaligned.txt $BASE_DIR/
fi
rm -rf $BASE_DIR/mfa_outputs_tmp
{
cd $BASE_DIR/mfa_outputs;
for file in *; do mv "$file" "${file#*_}"; done
}
