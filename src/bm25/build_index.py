import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="data/miracl_ar/corpus.jsonl")
    parser.add_argument("--index_dir", default="data/indexes/miracl_ar_bm25")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.index_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", os.path.dirname(args.collection),
        "--index", args.index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", str(args.threads),
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()
