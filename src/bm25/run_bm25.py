import os
import csv
import argparse
import subprocess
import ir_datasets

def write_queries_tsv(dataset_id: str, out_path: str):
    ds = ir_datasets.load(dataset_id)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for q in ds.queries_iter():
            w.writerow([q.query_id, q.text])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="miracl/ar/dev")
    parser.add_argument("--index_dir", default="data/indexes/miracl_ar_bm25")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--out_run", default="results/bm25/miracl_ar_dev.run")
    args = parser.parse_args()

    queries_tsv = "data/miracl_ar/queries_dev.tsv"
    write_queries_tsv(args.dataset, queries_tsv)

    os.makedirs(os.path.dirname(args.out_run), exist_ok=True)

    cmd = [
        "python", "-m", "pyserini.search.lucene",
        "--index", args.index_dir,
        "--topics", queries_tsv,
        "--topics-format", "tsv",
        "--output", args.out_run,
        "--hits", str(args.k),
        "--bm25"
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    print(f"✅ Wrote run file to {args.out_run}")

if __name__ == "__main__":
    main()
