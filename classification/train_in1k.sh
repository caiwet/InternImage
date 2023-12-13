#!/usr/bin/env bash

set -x

CONFIG=$1
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    srun -t 15:00:00 \
    --gres=gpu:${GPUS_PER_NODE} \
    --partition gpu_requeue \
    -c 1 --mem=300G --pty --account=rajpurkar_prr712 \
    ${SRUN_ARGS} \
    python -u main.py \
    --cfg ${CONFIG} \
    --accumulation-steps 1 \
    --local_rank 0 \
    --data-path /n/data1/hms/dbmi/rajpurkar/lab/MAIDA_ETT/all_data_split \
    --output cls_train_auc_08 ${@:4}
