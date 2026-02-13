import argparse
import ir_datasets
from ranx import Qrels, Run, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="miracl/ar/dev")
    parser.add_argument("--run_path", required=True)
    args = parser.parse_args()

    ds = ir_datasets.load(args.dataset)

    # ranx expects: {qid: {docid: relevance}}
    qrels_dict = {}
    for r in ds.qrels_iter():
        qrels_dict.setdefault(r.query_id, {})[r.doc_id] = int(r.relevance)

    qrels = Qrels(qrels_dict)
    run = Run.from_file(args.run_path, kind="trec")

    metrics = ["map", "mrr", "ndcg@10", "recall@100"]
    scores = evaluate(qrels, run, metrics)

    print("Dataset:", args.dataset)
    print("Run:", args.run_path)
    for m in metrics:
        print(f"{m}: {scores[m]:.6f}")


if __name__ == "__main__":
    main()
