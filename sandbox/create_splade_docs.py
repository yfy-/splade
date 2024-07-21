#!/usr/bin/env python3

import argparse
from collections import defaultdict
import logging
from pathlib import Path
from functools import partial

import numpy as np

import torch
from tqdm import tqdm

from splade_utils import (
    splade_encode,
    load_collection,
    load_tokenizer_and_model,
    quantize_sparse_8bit,
    recreate_text,
)

batch_path_fmt = "splade_encode_{batch_idx}.npz"


def load_batch(batch_dir, batch_idx) -> torch.Tensor:
    batch_path = batch_dir / batch_path_fmt.format(batch_idx=batch_idx)
    logging.info(f"Loading: {batch_idx}")
    return torch.Tensor(np.load(batch_path)["arr"])


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
    num_batches = np.ceil(num_doc / args.batch_size)
    logging.info(f"Collection size: {len(collection)}")
    work_dir = Path(args.work_dir)
    max_weight_path = work_dir / "global_max_weight.pt"
    if not args.skip_encode:
        global_max_weight = 0.0
        if max_weight_path.exists():
            global_max_weight = torch.load(max_weight_path)

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
            np.savez_compressed(
                work_dir / batch_path_fmt.format(batch_idx=i), arr=as_np
            )
            torch.save(global_max_weight, max_weight_path)

    # global_max_weight = torch.load(max_weight_path)
    if args.doc_compose:
        docid2docs = torch.load(args.doc_compose)
        docids = [str(did) for did in range(docid2docs.shape[0])]
    else:
        docid2docs = torch.arange(num_doc).reshape(num_doc, 1)
        docids = list(collection.keys())

    docid2idx = {docid: i for i, docid in enumerate(collection.keys())}
    rlookup = defaultdict(list)
    for did, compose_dids in enumerate(docid2docs):
        for cd in compose_dids:
            cd = cd.item()
            if cd == -1:
                break

            rlookup[docid2idx[cd]].append(did)

    composed_docs = torch.empty((len(docid2docs), len(tokenizer.vocab)))
    last_loaded_batch_idx = 0
    splade_encode_batch = load_batch(work_dir, last_loaded_batch_idx)
    total_proc = 0
    for i in tqdm(range(num_doc), desc="Doc compose"):
        encode_batch_idx = i - total_proc
        if encode_batch_idx == splade_encode_batch.shape[0]:
            total_proc += splade_encode_batch.shape[0]
            last_loaded_batch_idx += 1
            splade_encode_batch = load_batch(work_dir, last_loaded_batch_idx)
            encode_batch_idx = i - total_proc

        for did in rlookup[i]:
            logging.info(f"Adding {i} to {did}.")
            composed_docs[did] += splade_encode_batch[encode_batch_idx].clone()

    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    logging.info("Quantizing..")
    quantized = quantize_sparse_8bit(
        composed_docs, glob_max=torch.max(composed_docs)
    )
    del composed_docs
    quantized_batches = np.array_split(quantized, args.jobs)
    recreate_text_partial = partial(recreate_text, reverse_voc=reverse_voc)
    if args.jobs > 1:
        with torch.multiprocessing.get_context("spawn").Pool(args.jobs) as pool:
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
