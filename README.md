# Arabic Contriever IR

Zero-shot and fine-tuned dense retrieval evaluation on **MIRACL Arabic**, with a BM25 baseline.

## Results

Evaluated on **MIRACL Arabic dev** (2,896 queries, 2,061,414 documents).

| Model | MAP | MRR | nDCG@10 | Recall@100 |
|---|---|---|---|---|
| Contriever zero-shot | 0.0002 | 0.0005 | 0.0002 | 0.0019 |
| mContriever zero-shot | 0.1735 | 0.2263 | 0.2092 | 0.5314 |
| Contriever fine-tuned | 0.2706 | 0.3470 | 0.3190 | 0.6208 |
| BM25 (k1=0.9, b=0.4) | 0.3783 | 0.4588 | 0.4321 | 0.7859 |
| **mContriever fine-tuned** | **0.4180** | **0.4889** | **0.4864** | **0.9020** |

### Key Findings

1. **Multilingual pretraining is critical for Arabic** — mContriever zero-shot (0.209 nDCG@10) massively outperforms English Contriever zero-shot (0.000).
2. **Fine-tuning on a small dataset works** — 6,217 training examples lifted mContriever from 0.209 → 0.486 nDCG@10 (+133%).
3. **Fine-tuned mContriever surpasses BM25** — nDCG@10: +12.5%, MAP: +10.6%, Recall@100: +14.8%.
4. **Starting from the right base model matters** — fine-tuned English Contriever (0.319) still couldn't beat BM25.

### Fine-tuning Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `facebook/mcontriever` (or `facebook/contriever`) |
| Training data | MIRACL Arabic train split (3,495 queries, 6,217 triples) |
| Epochs | 3 |
| Batch size | 4 (effective 32 via grad_accum_steps=8) |
| Learning rate | 2e-5 (AdamW, linear warmup 10%) |
| Temperature (τ) | 0.05 |
| Loss | InfoNCE with in-batch + hard negatives |
| Negatives per query | 3 (random sampling) |
| Max sequence length | 256 |
| Mixed precision | FP16 |

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

### 3. Zero-shot dense retrieval

```bash
# Contriever (English)
python -m src.dense.run_contriever_zeroshot \
    --model_name facebook/contriever \
    --out_run results/dense/contriever_zero_miracl_ar_dev.run \
    --fp16 --batch_size 128

# mContriever (multilingual)
python -m src.dense.run_contriever_zeroshot \
    --model_name facebook/mcontriever \
    --out_run results/dense/mcontriever_zero_miracl_ar_dev.run \
    --fp16 --batch_size 128

# Evaluate
python -m src.eval.evaluate_run_ranx --run_path results/dense/contriever_zero_miracl_ar_dev.run
python -m src.eval.evaluate_run_ranx --run_path results/dense/mcontriever_zero_miracl_ar_dev.run
```

### 4. Fine-tuned dense retrieval

```bash
# Step 1 – Prepare training triples from MIRACL Arabic train split
python -m src.dense.prepare_training_data \
    --dataset miracl/ar/train \
    --neg_per_query 3

# Step 2 – Fine-tune
python -m src.dense.finetune_contriever \
    --model_name facebook/mcontriever \
    --train_data data/training/miracl_ar_train.jsonl \
    --epochs 3 --batch_size 4 --grad_accum_steps 8 \
    --lr 2e-5 --temperature 0.05 --fp16 --neg_per_query 3 \
    --out_dir models/mcontriever_finetuned_ar

# Step 3 – Retrieve with fine-tuned model
python -m src.dense.run_contriever_zeroshot \
    --model_name models/mcontriever_finetuned_ar \
    --out_run results/dense/mcontriever_ft_miracl_ar_dev.run \
    --fp16 --batch_size 128

# Step 4 – Evaluate
python -m src.eval.evaluate_run_ranx \
    --run_path results/dense/mcontriever_ft_miracl_ar_dev.run
```

> **Tip:** Use `--emb_cache_dir data/embeddings/<model>` to cache document
> embeddings to disk and skip re-encoding on reruns.

## Project Structure

```
src/
├── bm25/
│   ├── build_index.py          # Build Lucene index for BM25
│   └── run_bm25.py             # Run BM25 retrieval
├── data/
│   └── export_miracl_ar_to_jsonl.py  # Export MIRACL corpus to JSONL
├── dense/
│   ├── run_contriever_zeroshot.py    # Dense retrieval (zero-shot or fine-tuned)
│   ├── prepare_training_data.py      # Build contrastive training triples
│   └── finetune_contriever.py        # Fine-tune with InfoNCE loss
└── eval/
    └── evaluate_run_ranx.py          # Evaluate TREC run files
```
