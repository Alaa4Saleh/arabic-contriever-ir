# Arabic Contriever IR

Zero-shot and fine-tuned dense retrieval evaluation on **MIRACL Arabic**, with a BM25 baseline.

## Current Results

| Model | MAP | MRR | nDCG@10 | Recall@100 |
|---|---|---|---|---|
| BM25 (k1=0.9, b=0.4) | 0.3783 | 0.4588 | 0.4321 | 0.7859 |
| Contriever zero-shot | — | — | — | — |
| **Contriever fine-tuned** | — | — | — | — |

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Export MIRACL Arabic corpus

```bash
python -m src.data.export_miracl_ar_to_jsonl
```

### 2. BM25 baseline

```bash
# Build Lucene index
python -m src.bm25.build_index

# Run BM25 search
python -m src.bm25.run_bm25

# Evaluate
python -m src.eval.evaluate_run_ranx --run_path results/bm25/miracl_ar_dev.run
```

### 3. Zero-shot Contriever

```bash
python -m src.dense.run_contriever_zeroshot
python -m src.eval.evaluate_run_ranx --run_path results/dense/contriever_zero_miracl_ar_dev.run
```

### 4. Fine-tuned Contriever

```bash
# Step 1 – Prepare training triples from MIRACL Arabic train split
python -m src.dense.prepare_training_data \
    --dataset miracl/ar/train \
    --neg_per_query 7

# Step 2 – Fine-tune (small run: 3 epochs, batch 32)
python -m src.dense.finetune_contriever \
    --train_data data/training/miracl_ar_train.jsonl \
    --epochs 3 --batch_size 32 --lr 2e-5 \
    --temperature 0.05 \
    --out_dir models/contriever_finetuned_ar

# Step 3 – Retrieve with fine-tuned model (reuses zero-shot script)
python -m src.dense.run_contriever_zeroshot \
    --model_name models/contriever_finetuned_ar \
    --out_run results/dense/contriever_ft_miracl_ar_dev.run

# Step 4 – Evaluate
python -m src.eval.evaluate_run_ranx \
    --run_path results/dense/contriever_ft_miracl_ar_dev.run
```

> **Tip:** For a quick sanity check add `--max_steps 200` to fine-tuning and
> `--max_docs 50000` to retrieval.
