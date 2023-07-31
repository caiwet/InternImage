#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
GPU_ID=$3
WORK_DIR=$4

PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=63667 \
    $(dirname "$0")/train.py --checkpoint=$CONFIG --work-dir=${WORK_DIR} --gpu-id=$GPU_ID  --launcher pytorch ${@:3}