#!/bin/bash
# setup_and_train.sh — Fixes deps and runs fine-tuning using the conda env Python.
# Usage:  bash setup_and_train.sh
set -e

PROJ_DIR=~/projects/arabic-contriever-ir
PY=/anaconda/envs/py38_default/bin/python
PIP=/anaconda/envs/py38_default/bin/pip

echo "=== Step 1: Install compatible transformers ==="
$PIP install transformers==4.44.2
$PIP install faiss-cpu numpy tqdm ir_datasets

echo ""
echo "=== Step 2: Verify CUDA ==="
$PY -c "
import torch
print(f'torch version      : {torch.__version__}')
print(f'CUDA available     : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU                : {torch.cuda.get_device_name(0)}')
    t = torch.zeros(1, device='cuda')
    print(f'GPU tensor test    : OK')
else:
    print('WARNING: No GPU detected')

import transformers
print(f'transformers ver   : {transformers.__version__}')
"

echo ""
echo "=== Step 3: Run fine-tuning ==="
cd "$PROJ_DIR"
$PY -m src.dense.finetune_contriever \
    --train_data data/training/miracl_ar_train.jsonl \
    --epochs 3 --batch_size 8 --grad_accum_steps 4 \
    --lr 2e-5 --temperature 0.05 \
    --fp16 \
    --out_dir models/contriever_finetuned_ar

echo ""
echo "=== Done! Model saved to models/contriever_finetuned_ar ==="
