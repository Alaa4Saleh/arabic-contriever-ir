import os
import json
import argparse
import ir_datasets
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="miracl/ar/dev")
    parser.add_argument("--out_dir", default="data/miracl_ar")
    parser.add_argument("--max_docs", type=int, default=0, help="0 = export all docs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "corpus.jsonl")

    ds = ir_datasets.load(args.dataset)

    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for d in tqdm(ds.docs_iter(), desc="Exporting docs"):
            doc_id = d.doc_id
            title = (d.title or "").strip()
            text = (d.text or "").strip()
            contents = (title + "\n" + text).strip() if title else text

            rec = {"id": doc_id, "contents": contents}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            n += 1
            if args.max_docs and n >= args.max_docs:
                break

    print(f"✅ Wrote {n:,} docs to {out_path}")

if __name__ == "__main__":
    main()
