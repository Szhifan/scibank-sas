#!/usr/bin/env bash

ROOT=$(git rev-parse --show-toplevel)
RESULTS_ROOT="${ROOT}/results"

# mkdir -p ${RESULTS_ROOT}

### NAME YOUR EXPERIMENT HERE ##
EXP_NAME="gen"
################################

## Local variables for current experiment
EXP_ROOT="${RESULTS_ROOT}/${EXP_NAME}"

mkdir -p ${EXP_ROOT}3

#Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
python src/train_gen.py --save-dir "${EXP_ROOT}" \
    --model-name "t5-base" \
    --no-save \
    --batch-size 8 \
    --lr 2e-5 \
    --grad-accumulation 4 \
    --max-epoch 7 \
    --log-wandb \
    --freeze-layers 6 \
    --test-only \

            
               
