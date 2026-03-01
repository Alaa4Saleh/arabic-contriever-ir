import os
import argparse
import numpy as np
import ir_datasets
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
import faiss


def mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size=64, fp16=False):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tok = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
            out = model(**tok)
            emb = mean_pooling(out.last_hidden_state, tok["attention_mask"])
        emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
        all_embs.append(emb.cpu().numpy().astype("float32"))
    return np.vstack(all_embs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="miracl/ar/dev")
    parser.add_argument("--model_name", default="facebook/contriever")
    parser.add_argument("--out_run", default="results/dense/contriever_zero_miracl_ar_dev.run")
    parser.add_argument("--k", type=int, default=1000)

    parser.add_argument("--max_docs", type=int, default=0, help="0 = all docs; use e.g. 50000 for debug")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--index_type", choices=["flatip", "ivf"], default="flatip")
    parser.add_argument("--ivf_nlist", type=int, default=4096)
    parser.add_argument("--ivf_nprobe", type=int, default=16)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for faster encoding")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_run), exist_ok=True)

    if torch.cuda.is_available():
        try:
            # Test if the GPU actually works with this PyTorch build
            torch.zeros(1, device="cuda")
            device = torch.device("cuda")
        except Exception:
            print("⚠️  CUDA is available but GPU is incompatible with this PyTorch build. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    ds = ir_datasets.load(args.dataset)

    # 1) Load docs (IDs + texts)
    doc_ids = []
    doc_texts = []
    for d in tqdm(ds.docs_iter(), desc="Loading docs"):
        doc_ids.append(d.doc_id)
        title = (d.title or "").strip()
        text = (d.text or "").strip()
        contents = (title + "\n" + text).strip() if title else text
        doc_texts.append(contents)
        if args.max_docs and len(doc_ids) >= args.max_docs:
            break

    # 2) Encode docs
    dim = 768  # contriever base
    doc_embs = np.zeros((len(doc_texts), dim), dtype="float32")

    # encode in chunks to avoid huge temporary lists
    chunk = 10000
    for i in tqdm(range(0, len(doc_texts), chunk), desc="Encoding docs"):
        part = doc_texts[i:i + chunk]
        emb = encode_texts(part, tokenizer, model, device, batch_size=args.batch_size, fp16=args.fp16)
        doc_embs[i:i + emb.shape[0]] = emb

    # 3) Build FAISS index (cosine via inner product on normalized vectors)
    if args.index_type == "flatip":
        index = faiss.IndexFlatIP(dim)
    else:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, args.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(doc_embs)
        index.nprobe = args.ivf_nprobe

    index.add(doc_embs)

    # 4) Encode queries + retrieve
    queries = list(ds.queries_iter())
    with open(args.out_run, "w", encoding="utf-8") as f:
        for q in tqdm(queries, desc="Searching queries"):
            qid, qtext = q.query_id, q.text
            q_emb = encode_texts([qtext], tokenizer, model, device, batch_size=1, fp16=args.fp16)
            scores, idxs = index.search(q_emb, args.k)

            for rank, (doc_idx, score) in enumerate(zip(idxs[0], scores[0]), start=1):
                docid = doc_ids[int(doc_idx)]
                f.write(f"{qid} Q0 {docid} {rank} {float(score)} contriever_zero\n")

    print(f"✅ Wrote run file to {args.out_run}")


if __name__ == "__main__":
    main()
