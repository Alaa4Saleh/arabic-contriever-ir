"""
Prepare contrastive training data from MIRACL Arabic qrels.

For each query in the *train* split we collect:
  - positive passages  (relevance >= 1 in the qrels)
  - hard-negative passages (randomly sampled from the corpus, or BM25
    negatives when a BM25 run file is supplied).

Output: a JSONL file where every line is
  {"query": "...", "positive": "...", "negatives": ["...", ...]}

Usage
-----
# Minimal (random negatives):
python -m src.dense.prepare_training_data

# With BM25 hard negatives:
python -m src.dense.prepare_training_data \
    --bm25_run results/bm25/miracl_ar_train.run \
    --neg_per_query 7
"""

import os
import json
import random
import argparse
from collections import defaultdict

import ir_datasets
from tqdm import tqdm


def load_bm25_negatives(run_path: str, qrels_pos: dict, top_n: int = 100):
    """Return {qid: [docid, ...]} of BM25 hits that are NOT positive."""
    negatives: dict[str, list[str]] = defaultdict(list)
    with open(run_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, *_ = parts
            if qid in qrels_pos and docid not in qrels_pos[qid] and int(rank) <= top_n:
                negatives[qid].append(docid)
    return negatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="miracl/ar/train",
                        help="ir_datasets id (use train split for fine-tuning)")
    parser.add_argument("--out_path", default="data/training/miracl_ar_train.jsonl")
    parser.add_argument("--bm25_run", default=None,
                        help="Optional: path to BM25 run file for hard negatives")
    parser.add_argument("--neg_per_query", type=int, default=7,
                        help="Number of negatives per query-positive pair")
    parser.add_argument("--max_queries", type=int, default=0,
                        help="0 = use all queries; >0 to subsample for quick experiments")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)

    ds = ir_datasets.load(args.dataset)

    # ---- build doc lookup (id -> text) ---------------------------------
    print("Loading corpus …")
    doc_store: dict[str, str] = {}
    for d in tqdm(ds.docs_iter(), desc="docs"):
        title = (d.title or "").strip()
        text = (d.text or "").strip()
        contents = (title + "\n" + text).strip() if title else text
        doc_store[d.doc_id] = contents

    all_doc_ids = list(doc_store.keys())

    # ---- collect positive doc-ids per query ----------------------------
    qrels_pos: dict[str, set[str]] = defaultdict(set)
    for r in ds.qrels_iter():
        if int(r.relevance) >= 1:
            qrels_pos[r.query_id].add(r.doc_id)

    # ---- optional BM25 hard negatives ----------------------------------
    bm25_neg: dict[str, list[str]] = {}
    if args.bm25_run and os.path.exists(args.bm25_run):
        print(f"Loading BM25 hard negatives from {args.bm25_run}")
        bm25_neg = load_bm25_negatives(args.bm25_run, qrels_pos)

    # ---- build query lookup -------------------------------------------
    queries: dict[str, str] = {}
    for q in ds.queries_iter():
        queries[q.query_id] = q.text

    qids = sorted(qrels_pos.keys())
    if args.max_queries:
        random.shuffle(qids)
        qids = qids[:args.max_queries]

    # ---- write training JSONL ------------------------------------------
    n_written = 0
    with open(args.out_path, "w", encoding="utf-8") as f:
        for qid in tqdm(qids, desc="Building training pairs"):
            qtext = queries.get(qid)
            if qtext is None:
                continue

            pos_ids = list(qrels_pos[qid])

            for pid in pos_ids:
                pos_text = doc_store.get(pid)
                if pos_text is None:
                    continue

                # collect negatives
                neg_texts = []
                # prefer BM25 hard negatives
                if qid in bm25_neg:
                    pool = bm25_neg[qid]
                    chosen = random.sample(pool, min(args.neg_per_query, len(pool)))
                    neg_texts = [doc_store[did] for did in chosen if did in doc_store]

                # pad with random negatives if needed
                while len(neg_texts) < args.neg_per_query:
                    rid = random.choice(all_doc_ids)
                    if rid not in qrels_pos[qid]:
                        t = doc_store.get(rid)
                        if t:
                            neg_texts.append(t)

                record = {
                    "query": qtext,
                    "positive": pos_text,
                    "negatives": neg_texts[:args.neg_per_query],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"✅ Wrote {n_written:,} training examples to {args.out_path}")


if __name__ == "__main__":
    main()
