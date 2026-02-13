import os
import argparse

import ir_datasets
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="miracl/ar/dev")
    parser.add_argument("--index_dir", default="data/indexes/miracl_ar_bm25")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--out_run", default="results/bm25/miracl_ar_dev.run")
    # BM25 params (optional)
    parser.add_argument("--k1", type=float, default=0.9)
    parser.add_argument("--b", type=float, default=0.4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_run), exist_ok=True)

    ds = ir_datasets.load(args.dataset)
    searcher = LuceneSearcher(args.index_dir)
    searcher.set_bm25(args.k1, args.b)

    # Run file in TREC format: qid Q0 docid rank score tag
    tag = f"bm25_k1={args.k1}_b={args.b}"

    queries = list(ds.queries_iter())
    with open(args.out_run, "w", encoding="utf-8") as f:
        for q in tqdm(queries, desc=f"Searching {args.dataset}"):
            qid, text = q.query_id, q.text
            hits = searcher.search(text, k=args.k)

            for rank, h in enumerate(hits, start=1):
                # h.docid, h.score are provided by Pyserini
                f.write(f"{qid} Q0 {h.docid} {rank} {h.score} {tag}\n")

    print(f"✅ Wrote run file to {args.out_run} ({len(queries)} queries, top-{args.k})")


if __name__ == "__main__":
    main()
