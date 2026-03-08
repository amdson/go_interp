#!/usr/bin/env bash
# train.sh — Exact commands used to produce the five ESMC checkpoints in the paper.
#
# Run from go_ml/scripts/. Checkpoints are written to ../../checkpoints/.
# Lightning auto-increments the version suffix (-v1, -v2, ...) so the
# mapping below is only valid if the runs are executed in order on a clean
# output directory. The checkpoint → hyperparameter mapping is:
#
#   func_cond_finetune_esmc.ckpt     context=100  span=5   (base run)
#   func_cond_finetune_esmc-v1.ckpt  context=100  span=2
#   func_cond_finetune_esmc-v2.ckpt  context=100  span=10
#   func_cond_finetune_esmc-v3.ckpt  context=50   span=5
#   func_cond_finetune_esmc-v4.ckpt  context=200  span=5
#
# Hardware: single A100 80GB GPU, ~8 hours per run.
# Data:     ../../data/train_esm_datasets/{train,val}_dataset.pkl

set -euo pipefail

GPU=${1:-0}   # pass GPU id as first argument, default 0

echo "=== Run 1/5: context=100, span=5 (base checkpoint) ==="
python train_func_cond_esmc.py --gpu_id "$GPU" \
    --mask_func span --context_length 100 --span_mask_length 5

echo "=== Run 2/5: context=100, span=2 ==="
python train_func_cond_esmc.py --gpu_id "$GPU" \
    --mask_func span --context_length 100 --span_mask_length 2

echo "=== Run 3/5: context=100, span=10 ==="
python train_func_cond_esmc.py --gpu_id "$GPU" \
    --mask_func span --context_length 100 --span_mask_length 10

echo "=== Run 4/5: context=50, span=5 ==="
python train_func_cond_esmc.py --gpu_id "$GPU" \
    --mask_func span --context_length 50 --span_mask_length 5

echo "=== Run 5/5: context=200, span=5 ==="
python train_func_cond_esmc.py --gpu_id "$GPU" \
    --mask_func span --context_length 200 --span_mask_length 5

echo "All training runs complete."
