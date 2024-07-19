#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from functools import partial

import numpy as np

import torch

from splade_utils import (
    splade_encode,
    load_collection,
    load_tokenizer_and_model,
    quantize_sparse_8bit,
    recreate_text,
)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("collection")
    parser.add_argument("work_dir")
    parser.add_argument("model_name_or_dir")
    parser.add_argument("-b", "--batch-size", type=int, default=100000)
    parser.add_argument("-e", "--encode-batch-size", type=int, default=16)
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("-s", "--skip-encode", action="store_true")
    parser.add_argument("--encode-batch", type=int, default=0)
    parser.add_argument("-l", "--lora-model-dir")
    parser.add_argument("-d", "--doc-compose")
    args = parser.parse_args()
    lora_model_dir = args.lora_model_dir
    if lora_model_dir is not None:
        lora_model_dir = Path(lora_model_dir)
    tokenizer, splade = load_tokenizer_and_model(
        args.model_name_or_dir, lora_model_dir
    )
    collection = load_collection(args.collection)
    num_doc = len(collection)
    logging.info(f"Collection size: {len(collection)}")
    work_dir = Path(args.work_dir)
    max_weight_path = work_dir / "global_max_weight.pt"
    if not args.skip_encode:
        global_max_weight = 0.0
        if max_weight_path.exists():
            global_max_weight = torch.load(max_weight_path)

        num_batches = np.ceil(num_doc / args.batch_size)
        docs = list(collection.values())
        np_texts = np.array(docs, dtype="object")
        batches = np.array_split(np_texts, num_batches)
        if torch.cuda.is_available():
            logging.info("Moving model to cuda.")
            splade = splade.to("cuda")

        splade.eval()
        for i, batch in enumerate(batches):
            if i < args.encode_batch:
                continue

            logging.info(f"Encoding batch: {i}")
            batch_encoded = splade_encode(
                batch, splade, tokenizer, batch_size=args.encode_batch_size
            )
            global_max_weight = max(global_max_weight, torch.max(batch_encoded))
            as_np = (
                batch_encoded.detach()
                .cpu()
                .resolve_conj()
                .resolve_neg()
                .numpy()
            )
            np.savez_compressed(work_dir / f"splade_encode_{i}.npz", arr=as_np)
            torch.save(global_max_weight, max_weight_path)

    # global_max_weight = torch.load(max_weight_path)
    if args.doc_compose:
        docid2docs = torch.load(args.doc_compose)
        docids = [str(did) for did in range(docid2docs.shape[0])]
    else:
        docid2docs = torch.arange(num_doc).reshape(num_doc, 1)
        docids = list(collection.keys())

    loaded_batches = []
    batch_idx = 0
    while (batch_path := work_dir / f"splade_encode_{batch_idx}.npz").exists():
        logging.info(f"Loading: {batch_idx}")
        loaded_batches.append(torch.Tensor(np.load(batch_path)["arr"]))
        batch_idx += 1

    splade_encoded = torch.cat(loaded_batches, dim=0)
    composed_docs = torch.empty(
        (len(docid2docs), splade_encoded.shape[1]), dtype=splade_encoded.dtype
    )

    docid2idx = {docid: i for i, docid in enumerate(collection.keys())}
    for did, compose_dids in enumerate(docid2docs):
        new_doc_weights = None
        for cd in compose_dids:
            cd = cd.item()
            if cd == -1:
                break

            cd_weights = splade_encoded[docid2idx[cd]].clone()
            if new_doc_weights is None:
                new_doc_weights = cd_weights
            else:
                new_doc_weights += cd_weights

        composed_docs[did] = new_doc_weights

    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    logging.info("Quantizing..")
    quantized = quantize_sparse_8bit(
        composed_docs, glob_max=torch.max(composed_docs)
    )
    del composed_docs
    quantized_batches = np.array_split(quantized, args.jobs)
    recreate_text_partial = partial(recreate_text, reverse_voc=reverse_voc)
    if args.jobs > 1:
        with torch.multiprocessing.get_context("spawn").Pool(
            args.jobs
        ) as pool:
            recreated = pool.map(recreate_text_partial, quantized_batches)
    else:
        recreated = [recreate_text_partial(qb) for qb in quantized_batches]

    docidx = 0
    for qb in recreated:
        for d in qb:
            print(f"{docids[docidx]}\t{d}")
            docidx += 1

    return 0


if __name__ == "__main__":
    exit(main())
