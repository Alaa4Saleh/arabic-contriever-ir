# Technical Run Report

## Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA Tesla M60 (8 GB VRAM, Maxwell arch, compute capability 5.2) |
| NVIDIA Driver | 535.288.01 |
| CUDA Version | 12.2 |
| PyTorch | 2.5.1+cu124 (loaded from conda env) |
| Transformers | 4.44.2 |
| Python | 3.10 (`/anaconda/envs/py38_default/bin/python`) |
| FAISS | faiss-cpu (flat inner product index) |

---

## Dataset

| Property | Value |
|---|---|
| Corpus | MIRACL Arabic — 2,061,414 documents (Arabic Wikipedia) |
| Train queries | 3,495 (MIRACL `ar/train` split) |
| Train triples | 6,217 (query + positive + negatives) |
| Dev queries | 2,896 (MIRACL `ar/dev` split) |
| Negative sampling | Random from corpus (no BM25 hard negatives) |
| Negatives per query | 3 |

---

## Training Times

| Run | Duration (approx.) |
|---|---|
| Contriever fine-tuning (3 epochs) | ~2–3 hours |
| mContriever fine-tuning (3 epochs) | ~2–3 hours |

Both used identical hyperparameters:

| Parameter | Value |
|---|---|
| Batch size | 4 |
| Gradient accumulation steps | 8 |
| Effective batch size | 32 |
| Learning rate | 2e-5 |
| Optimizer | AdamW (weight_decay=0.01) |
| Scheduler | Linear warmup (10% of steps) + linear decay |
| Temperature (τ) | 0.05 |
| Max sequence length | 256 tokens |
| Loss | InfoNCE (in-batch negatives + 3 explicit negatives) |
| Negatives gradient | Detached (encoded with `torch.no_grad()` to save VRAM) |
| Mixed precision | FP16 |
| Total optimizer steps | ~582 per model |
| Warmup steps | ~58 per model |
| Seed | 42 |

---

## Inference Times (Document Encoding + FAISS Search)

| Run | Batch Size | FP16 | Duration (approx.) |
|---|---|---|---|
| Contriever zero-shot | 64 | No | ~16–18 hours |
| Contriever fine-tuned | 128 | Yes | ~16–18 hours |
| mContriever zero-shot | 128 | Yes | ~16–18 hours |
| mContriever fine-tuned | 128 | Yes | ~16–18 hours |

**Breakdown per run:**
- Document encoding: ~16–18 hours (2,061,414 docs × 12-layer BERT on Tesla M60)
- FAISS index build (FlatIP): < 1 minute
- Query encoding + search (2,896 queries): < 10 minutes
- Bottleneck: 100% document encoding

Note: The Contriever zero-shot run was the first and used batch_size=64 without FP16. All subsequent runs used batch_size=128 + FP16 after we added that optimization.

---