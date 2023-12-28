#!/bin/sh

export PYTHONPATH=./
eval "$(conda shell.bash hook)"
PYTHON=python3

TRAIN_CODE=train.py
TEST_CODE=test.py

dataset=DS-Net
exp_name=$1
config_name=$2

exp_dir=exp/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
config=config/${dataset}/${config_name}.yaml

mkdir -p ${model_dir} ${result_dir}
mkdir -p ${result_dir}/last
mkdir -p ${result_dir}/best
cp tool/train.sh tool/${TRAIN_CODE} tool/test.sh tool/${TEST_CODE} ${exp_dir}
cp models/architectures.py models/blocks.py ${exp_dir}
cp ${config} ${exp_dir}


now=$(date +"%Y%m%d_%H%M%S")
$PYTHON ${exp_dir}/${TRAIN_CODE} \
  --config=${config} \
  save_path ${exp_dir} \
  2>&1 | tee ${exp_dir}/train-$now.log
