"""
Fine-tune facebook/contriever on Arabic IR data with InfoNCE loss.

The script reads JSONL training data produced by prepare_training_data.py.
Each line: {"query": str, "positive": str, "negatives": [str, ...]}

Training uses *in-batch negatives* plus the explicit hard negatives:
  loss = -log( exp(sim(q,p+)/τ) / Σ exp(sim(q,pj)/τ) )

Usage
-----
python -m src.dense.finetune_contriever \
    --train_data data/training/miracl_ar_train.jsonl \
    --epochs 3 --lr 2e-5 --batch_size 32 --temperature 0.05 \
    --out_dir models/contriever_finetuned_ar
"""

import os
import json
import math
import random
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

class ContrastiveDataset(Dataset):
    """
    Each item returns (query, positive, negative_1, …, negative_n).
    """
    def __init__(self, path: str, neg_per_query: int = 7):
        self.samples = []
        self.neg_per_query = neg_per_query
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.samples.append(rec)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        negs = rec["negatives"][:self.neg_per_query]
        # pad if fewer negatives than expected
        while len(negs) < self.neg_per_query:
            negs.append("")
        return rec["query"], rec["positive"], negs


def collate_fn(batch):
    """Return (queries, positives, list_of_negatives)."""
    queries, positives, neg_lists = zip(*batch)
    return list(queries), list(positives), list(neg_lists)


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    In-batch negatives + explicit hard negatives.

    For a batch of B queries:
      q_emb:   (B, D)
      p_emb:   (B, D)   — positive for each query
      n_embs:  (B, N, D) — explicit negatives

    Similarity matrix is (B, B + B*N): each query scored against
    all positives (in-batch) and its own hard negatives.
    """
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, q_emb, p_emb, n_embs=None):
        # q_emb: (B, D), p_emb: (B, D)
        # in-batch scores
        scores = torch.mm(q_emb, p_emb.t()) / self.temperature  # (B, B)

        if n_embs is not None and n_embs.shape[1] > 0:
            # hard-negative scores  (B, N)
            neg_scores = torch.bmm(
                q_emb.unsqueeze(1), n_embs.transpose(1, 2)
            ).squeeze(1) / self.temperature
            scores = torch.cat([scores, neg_scores], dim=1)  # (B, B+N)

        labels = torch.arange(q_emb.size(0), device=q_emb.device)
        return F.cross_entropy(scores, labels)


# ---------------------------------------------------------------------------
# encoding helper
# ---------------------------------------------------------------------------

def encode(texts, tokenizer, model, device, max_length=256):
    tok = tokenizer(texts, padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt")
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model(**tok)
    emb = mean_pooling(out.last_hidden_state, tok["attention_mask"])
    return F.normalize(emb, p=2, dim=1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="data/training/miracl_ar_train.jsonl")
    parser.add_argument("--model_name", default="facebook/contriever")
    parser.add_argument("--out_dir", default="models/contriever_finetuned_ar")

    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--neg_per_query", type=int, default=7)
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (simulate larger batch)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed-precision training (needs CUDA)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Stop early after N optimizer steps (0 = full training)")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- device --------------------------------------------------------
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except Exception:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    use_amp = args.fp16 and device.type == "cuda"

    # ---- model & tokenizer ---------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.train()

    # ---- data ----------------------------------------------------------
    dataset = ContrastiveDataset(args.train_data, neg_per_query=args.neg_per_query)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True, num_workers=0)

    total_steps = (len(loader) * args.epochs) // args.grad_accum_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    print(f"Training samples : {len(dataset):,}")
    print(f"Batch size       : {args.batch_size} (× {args.grad_accum_steps} accum)")
    print(f"Total opt. steps : {total_steps:,}")
    print(f"Warmup steps     : {warmup_steps:,}")

    # ---- optimizer & scheduler -----------------------------------------
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                     if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                     if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = InfoNCELoss(temperature=args.temperature)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- training loop -------------------------------------------------
    global_step = 0
    running_loss = 0.0

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (queries, positives, neg_lists) in enumerate(pbar):
            # flatten negatives: list[list[str]] → list[str]
            flat_negs = [n for nlist in neg_lists for n in nlist]

            with torch.amp.autocast("cuda", enabled=use_amp):
                q_emb = encode(queries, tokenizer, model, device, args.max_length)
                p_emb = encode(positives, tokenizer, model, device, args.max_length)

                # Encode negatives WITHOUT gradients to save GPU memory
                if flat_negs and any(n != "" for n in flat_negs):
                    with torch.no_grad():
                        n_emb_parts = []
                        neg_chunk = args.batch_size  # encode negs in small chunks
                        for ni in range(0, len(flat_negs), neg_chunk):
                            part = flat_negs[ni:ni + neg_chunk]
                            n_emb_parts.append(
                                encode(part, tokenizer, model, device, args.max_length)
                            )
                        n_emb = torch.cat(n_emb_parts, dim=0)
                    B = q_emb.size(0)
                    N = args.neg_per_query
                    n_emb = n_emb.view(B, N, -1)
                else:
                    n_emb = None

                loss = criterion(q_emb, p_emb, n_emb)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                running_loss += loss.item() * args.grad_accum_steps

                if global_step % args.log_every == 0:
                    avg = running_loss / args.log_every
                    lr_now = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.2e}",
                                     step=global_step)
                    running_loss = 0.0

                if args.max_steps and global_step >= args.max_steps:
                    print(f"⏹  Reached max_steps={args.max_steps}; stopping.")
                    break

        if args.max_steps and global_step >= args.max_steps:
            break

    # ---- save ----------------------------------------------------------
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # save training args for reproducibility
    with open(os.path.join(args.out_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"✅ Saved fine-tuned model to {args.out_dir}")


if __name__ == "__main__":
    main()
