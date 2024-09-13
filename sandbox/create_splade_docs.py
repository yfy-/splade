#!/usr/bin/env python3

import argparse
from collections import defaultdict
import logging
import os.path
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
    parser.add_argument("compose_cache_path")
    parser.add_argument("-b", "--batch-size", type=int, default=100000)
    parser.add_argument("-e", "--encode-batch-size", type=int, default=16)
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("-s", "--skip-encode", action="store_true")
    parser.add_argument("--encode-batch", type=int, default=0)
    parser.add_argument("-l", "--lora-model-dir")
    parser.add_argument("-d", "--doc-compose")
    parser.add_argument("-n", "--nearest-percent", type=float, default=1.0)
    parser.add_argument("-p", "--pruning-submass", type=float, default=1.0)
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

    if args.doc_compose:
        docid2docs = torch.load(args.doc_compose, map_location="cpu")
        docids = [str(did) for did in range(docid2docs.shape[0])]
    else:
        docid2docs = torch.arange(num_doc).reshape(num_doc, 1)
        docids = list(collection.keys())

    if not os.path.exists(args.compose_cache_path):
        # Take nearest percent of passages for each centroid
        if args.nearest_percent < 1.0:
            psg_counts = torch.sum(docid2docs != -1, dim=1, keepdim=True)
            psg_counts_keep = torch.ceil(psg_counts * args.nearest_percent).int()
            remove_mask = (
                torch.arange(docid2docs.shape[1]).repeat(docid2docs.shape[0], 1)
                >= psg_counts_keep
            )
            docid2docs[remove_mask] = -1.0

        # Create a lookup dict where keys are passages and values are a
        # list of centroids they appear. This way we can load splade
        # vector batches in sequantial order to not blow up the memory.
        docid2idx = {docid: i for i, docid in enumerate(collection.keys())}
        rlookup = defaultdict(list)
        for did, compose_dids in tqdm(
            enumerate(docid2docs),
            desc="Build reverse lookup",
            total=len(docid2docs),
        ):
            for cd in compose_dids:
                cd = cd.item()
                if cd == -1:
                    break

                rlookup[docid2idx[cd]].append(did)

        # Take average of splade encoding for each centroid by loading
        # them batch by batch.
        composed_docs = torch.empty((len(docid2docs), len(tokenizer.vocab)))
        num_psg_cent = torch.zeros((len(docid2docs), 1))
        del docid2docs
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
                composed_docs[did] += splade_encode_batch[encode_batch_idx].clone()
                num_psg_cent[did] += 1

        del rlookup
        composed_docs /= num_psg_cent
        torch.save(composed_docs, args.compose_cache_path)
    else:
        del docid2docs
        composed_docs = torch.load(args.compose_cache_path, map_location="cpu")

    # Prune each centroid vector to its specified L1 submass.
    if args.pruning_submass < 1.0:
        logging.info(f"Pruning submass factor: {args.pruning_submass}")
        tot_set = 0
        for i in tqdm(
            range(len(composed_docs)), total=len(composed_docs), desc="Pruning"
        ):
            doc_sorted, sort_idx = torch.sort(composed_docs[i], descending=True)
            doc_cumsum = torch.cumsum(doc_sorted, dim=0)
            # Normalize with largest num
            doc_cumsum /= doc_cumsum[-1:].clone()
            search_idx = torch.searchsorted(doc_cumsum, args.pruning_submass)
            keep_mask = torch.arange(len(sort_idx)) <= search_idx

            prune_mask = torch.ones(len(sort_idx), dtype=torch.bool)

            # Set most important dimensions to False so that we do not set
            # them to 0.0.
            prune_mask[sort_idx[keep_mask]] = False
            composed_docs[i][prune_mask] = 0.0
            tot_set += torch.sum(composed_docs[i] > 0.0).item()

        logging.info(f"Avg dim set: {tot_set / len(composed_docs)}")

    # Quantize and then reconstruct textual documents.
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    logging.info("Quantizing..")
    quantized = quantize_sparse_8bit(
        composed_docs, glob_max=torch.max(composed_docs)
    )
    del composed_docs
    docidx = 0
    for q_split in quantized.split(10000):
        quantized_batches = np.array_split(q_split, args.jobs)
        recreate_text_partial = partial(recreate_text, reverse_voc=reverse_voc)
        if args.jobs > 1:
            with torch.multiprocessing.get_context("spawn").Pool(
                args.jobs
            ) as pool:
                recreated = pool.map(recreate_text_partial, quantized_batches)
        else:
            recreated = [recreate_text_partial(qb) for qb in quantized_batches]

        for qb in recreated:
            for d in qb:
                print(f"{docids[docidx]}\t", end="")
                for term, weight in d.items():
                    for _ in range(weight):
                        print(f"{term} ", end="")
                print("")
                docidx += 1
    return 0


if __name__ == "__main__":
    exit(main())
