#!/bin/sh

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
PYTHON=python

TEST_CODE=test.py

dataset=DS-Net
exp_name=$1
config_name=$2
exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
# config=config/${dataset}/${dataset}_${exp_name}.yaml
config=${exp_dir}/${config_name}.yaml

mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best

now=$(date +"%Y%m%d_%H%M%S")
# cp ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}
cp tool/test.sh tool/${TEST_CODE} ${exp_dir}

#: '
$PYTHON -u ${exp_dir}/${TEST_CODE} \
  --config=${config} \
  save_folder ${result_dir}/best \
  model_path ${model_dir}/model_best.pth \
  2>&1 | tee ${exp_dir}/test_best-$now.log
#'

#: '
# $PYTHON -u ${exp_dir}/${TEST_CODE} \
#   --config=${config} \
#   save_folder ${result_dir}/last \
#   model_path ${model_dir}/model_last.pth \
#   2>&1 | tee ${exp_dir}/test_last-$now.log
#'
