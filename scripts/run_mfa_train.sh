#!/bin/bash
set -e

NUM_JOB=${NUM_JOB:-36}
MFA_VERSION=${MFA_VERSION:-"1"}
echo "| Training MFA using ${NUM_JOB} cores."
BASE_DIR=data/processed/$CORPUS
rm -rf $BASE_DIR/mfa_outputs_tmp

# train mfa
mfa train $BASE_DIR/mfa_inputs $BASE_DIR/mfa_dict.txt $BASE_DIR/mfa_outputs_tmp -t $BASE_DIR/mfa_tmp -o $BASE_DIR/mfa_model.zip --clean -j $NUM_JOB --config_path data_gen/tts/mfa_config.yaml