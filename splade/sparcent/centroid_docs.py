#!/usr/bin/env python3

"""Create splade docs in text format.

Each output document is the average SPLADE vector of the passages composing it
(e.g. the passages of a ColBERT centroid), optionally pruned to an L1 submass,
written as terms repeated by their quantized weights.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch
from psutil import virtual_memory
from tqdm import tqdm

from splade.models.transformer_rep import Splade
from splade.sparcent.disk_loader import BatchSparseTensorDiskLoader
from splade.sparcent.utils import (load_collection, load_tokenizer_and_model,
                                   quantize_sparse_8bit, recreate_text,
                                   splade_encode)

COMPOSE_CHECKPOINT = "compose-checkpoint"


def batch_path(work_dir: Path, batch_idx: int) -> Path:
    return work_dir / f"splade_encode_{batch_idx}.pt"


def load_batch(work_dir: Path, batch_idx: int) -> torch.Tensor:
    logging.info(f"Loading: {batch_idx}")
    return torch.load(batch_path(work_dir, batch_idx)).to_dense()


def encode_collection(
    collection: Dict[int, str],
    splade: Splade,
    tokenizer,
    work_dir: Path,
    batch_size=100000,
    encode_batch_size=16,
    start_batch=0,
):
    """Encode the collection with SPLADE and save each batch as a sparse tensor."""
    work_dir.mkdir(parents=True, exist_ok=True)
    max_weight_path = work_dir / "global_max_weight.pt"
    global_max_weight = 0.0
    if max_weight_path.exists():
        global_max_weight = torch.load(max_weight_path)

    num_batches = np.ceil(len(collection) / batch_size)
    np_texts = np.array(list(collection.values()), dtype="object")
    batches = np.array_split(np_texts, num_batches)
    if torch.cuda.is_available():
        logging.info("Moving model to cuda.")
        splade = splade.to("cuda")

    splade.eval()
    for i, batch in enumerate(batches):
        if i < start_batch:
            continue

        logging.info(f"Encoding batch: {i}")
        batch_encoded = splade_encode(
            batch, splade, tokenizer, batch_size=encode_batch_size
        )
        global_max_weight = max(global_max_weight, torch.max(batch_encoded))
        # Rename after saving so an interrupted run never leaves a partial batch.
        tmp_path = batch_path(work_dir, i).with_suffix(".tmp")
        torch.save(batch_encoded.to_sparse(), tmp_path)
        tmp_path.rename(batch_path(work_dir, i))
        torch.save(global_max_weight, max_weight_path)


def load_doc_compose(
    doc_compose: Optional[str], collection: Dict[int, str]
) -> Tuple[torch.Tensor, List]:
    """Load the passages composing each document and the document ids.

    Without `doc_compose`, every passage composes a document by itself.
    """
    if doc_compose:
        docid2docs = torch.load(doc_compose, map_location="cpu")
        docids = [str(did) for did in range(docid2docs.shape[0])]
    else:
        num_doc = len(collection)
        docid2docs = torch.arange(num_doc).reshape(num_doc, 1)
        docids = list(collection.keys())

    return docid2docs, docids


def keep_nearest(
    docid2docs: torch.Tensor, nearest_percent: float
) -> torch.Tensor:
    """Keep only the nearest percent of passages for each document."""
    psg_counts = torch.sum(docid2docs != -1, dim=1, keepdim=True)
    psg_counts_keep = torch.ceil(psg_counts * nearest_percent).int()
    remove_mask = (
        torch.arange(docid2docs.shape[1]).repeat(docid2docs.shape[0], 1)
        >= psg_counts_keep
    )
    docid2docs[remove_mask] = -1.0
    return docid2docs


def build_reverse_lookup(
    docid2docs: torch.Tensor, collection: Dict[int, str]
) -> Dict[int, List[int]]:
    """Map passages (by collection position) to the documents they compose.

    This way we can load splade vector batches in sequential order to not
    blow up the memory.
    """
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

    return rlookup


def compose_vectors(
    composed_docs_loader: BatchSparseTensorDiskLoader,
    rlookup: Dict[int, List[int]],
    num_doc: int,
    work_dir: Path,
):
    """Average the splade vectors of the passages composing each document."""
    num_psg_cent = torch.zeros((composed_docs_loader.shape[0], 1))
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

        for did in rlookup.get(i, ()):
            composed_docs, cdi = composed_docs_loader[did]
            composed_docs[cdi] += splade_encode_batch[encode_batch_idx].clone()
            num_psg_cent[did] += 1

    # Documents without passages stay empty instead of becoming NaN.
    num_psg_cent[num_psg_cent == 0] = 1
    c_batch_size = composed_docs_loader.batch_size
    i = 0
    for composed_docs in composed_docs_loader.batch_iter():
        composed_docs /= num_psg_cent[i : i + c_batch_size]
        i += c_batch_size


def prune_submass(
    composed_docs_loader: BatchSparseTensorDiskLoader, pruning_submass: float
):
    """Prune each document vector to its specified L1 submass."""
    logging.info(f"Pruning submass factor: {pruning_submass}")
    tot_set = 0
    for i in tqdm(
        range(composed_docs_loader.shape[0]),
        total=composed_docs_loader.shape[0],
        desc="Pruning",
    ):
        composed_docs, cdi = composed_docs_loader[i]
        doc_sorted, sort_idx = torch.sort(composed_docs[cdi], descending=True)
        doc_cumsum = torch.cumsum(doc_sorted, dim=0)
        if doc_cumsum[-1] == 0.0:
            continue

        # Normalize with largest num
        doc_cumsum /= doc_cumsum[-1:].clone()
        search_idx = torch.searchsorted(doc_cumsum, pruning_submass)
        keep_mask = torch.arange(len(sort_idx)) <= search_idx

        prune_mask = torch.ones(len(sort_idx), dtype=torch.bool)

        # Set most important dimensions to False so that
        # we do not set them to 0.0.
        prune_mask[sort_idx[keep_mask]] = False
        composed_docs[cdi][prune_mask] = 0.0
        tot_set += torch.sum(composed_docs[cdi] > 0.0).item()

    logging.info(f"Avg dim set: {tot_set / composed_docs_loader.shape[0]}")


def recreate_split(
    split: torch.Tensor, reverse_voc: Dict[int, str], jobs: int
) -> List[Dict]:
    """Recreate the term weights of a split of vectors with `jobs` processes."""
    split_batches = np.array_split(split, jobs)
    recreate_text_partial = partial(recreate_text, reverse_voc=reverse_voc)
    if jobs > 1:
        with torch.multiprocessing.get_context("spawn").Pool(jobs) as pool:
            recreated = pool.map(recreate_text_partial, split_batches)
    else:
        recreated = [recreate_text_partial(sb) for sb in split_batches]

    return [d for batch in recreated for d in batch]


def write_text_docs(
    composed_docs_loader: BatchSparseTensorDiskLoader,
    docids: List,
    reverse_voc: Dict[int, str],
    jobs: int,
    out: TextIO,
):
    """Quantize the documents to 8 bits and write terms repeated by weight."""
    logging.info("Quantizing..")
    composed_docs_max = 0.0
    for composed_docs in composed_docs_loader.batch_iter(immut=True):
        composed_docs_max = max(
            torch.max(composed_docs).item(), composed_docs_max
        )

    docidx = 0
    for composed_docs in composed_docs_loader.batch_iter(immut=True):
        quantized = quantize_sparse_8bit(
            composed_docs, glob_max=composed_docs_max
        )
        for q_split in quantized.split(5000):
            for d in recreate_split(q_split, reverse_voc, jobs):
                terms = "".join(
                    f"{term} " * weight for term, weight in d.items()
                )
                out.write(f"{docids[docidx]}\t{terms}\n")
                docidx += 1


def write_jsonl_docs(
    composed_docs_loader: BatchSparseTensorDiskLoader,
    docids: List,
    collection: Dict[int, str],
    reverse_voc: Dict[int, str],
    jobs: int,
    out: TextIO,
):
    """Write the documents as JSON lines with non-quantized term weights."""
    docidx = 0
    for composed_docs in composed_docs_loader.batch_iter(immut=True):
        for split in composed_docs.split(5000):
            for d in recreate_split(split, reverse_voc, jobs):
                jsonl_dict = {
                    "id": docids[docidx],
                    "content": collection[docids[docidx]],
                    "vector": d,
                }
                out.write(json.dumps(jsonl_dict) + "\n")
                docidx += 1


def create_centroid_docs(
    collection_path: str,
    work_dir: Path,
    model_name_or_dir: str,
    compose_cache_path: Path,
    out: TextIO,
    batch_size=100000,
    encode_batch_size=16,
    jobs=1,
    skip_encode=False,
    encode_batch=0,
    lora_model_dir: Optional[Path] = None,
    doc_compose: Optional[str] = None,
    nearest_percent=1.0,
    pruning_submass=1.0,
    jsonl=False,
):
    """Encode the collection (unless skipped) and write the composed docs."""
    tokenizer, splade = load_tokenizer_and_model(
        model_name_or_dir, lora_model_dir
    )
    collection = load_collection(collection_path)
    num_doc = len(collection)
    logging.info(f"Collection size: {num_doc}")
    work_dir = Path(work_dir)
    if not skip_encode:
        encode_collection(
            collection,
            splade,
            tokenizer,
            work_dir,
            batch_size=batch_size,
            encode_batch_size=encode_batch_size,
            start_batch=encode_batch,
        )

    docid2docs, docids = load_doc_compose(doc_compose, collection)

    # Take average of splade encoding for each document by loading them
    # batch by batch.
    mem_limit = virtual_memory().total // 4
    composed_shape = (len(docid2docs), len(tokenizer.vocab))
    compose_batch_size = mem_limit // (composed_shape[1] * 4)
    logging.info(f"Composed batch_size={compose_batch_size}")
    with BatchSparseTensorDiskLoader(
        Path(compose_cache_path),
        composed_shape,
        batch_size=compose_batch_size,
    ) as composed_docs_loader:
        if not composed_docs_loader.load_from_checkpoint(COMPOSE_CHECKPOINT):
            if nearest_percent < 1.0:
                docid2docs = keep_nearest(docid2docs, nearest_percent)

            rlookup = build_reverse_lookup(docid2docs, collection)
            del docid2docs
            compose_vectors(composed_docs_loader, rlookup, num_doc, work_dir)
            del rlookup
            composed_docs_loader.checkpoint(COMPOSE_CHECKPOINT)
        else:
            del docid2docs

        if pruning_submass < 1.0:
            prune_submass(composed_docs_loader, pruning_submass)

        reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
        if jsonl:
            write_jsonl_docs(
                composed_docs_loader,
                docids,
                collection,
                reverse_voc,
                jobs,
                out,
            )
        else:
            write_text_docs(
                composed_docs_loader, docids, reverse_voc, jobs, out
            )


def open_output(path: Optional[str]) -> TextIO:
    if path is None:
        return sys.stdout

    return gzip.open(path, "wt") if path.endswith(".gz") else open(path, "w")


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
    parser.add_argument("--jsonl", action="store_true")
    parser.add_argument(
        "-o",
        "--output",
        help="Output file (.gz is compressed), default stdout.",
    )
    args = parser.parse_args()
    lora_model_dir = args.lora_model_dir
    if lora_model_dir is not None:
        lora_model_dir = Path(lora_model_dir)

    out = open_output(args.output)
    try:
        create_centroid_docs(
            args.collection,
            Path(args.work_dir),
            args.model_name_or_dir,
            Path(args.compose_cache_path),
            out,
            batch_size=args.batch_size,
            encode_batch_size=args.encode_batch_size,
            jobs=args.jobs,
            skip_encode=args.skip_encode,
            encode_batch=args.encode_batch,
            lora_model_dir=lora_model_dir,
            doc_compose=args.doc_compose,
            nearest_percent=args.nearest_percent,
            pruning_submass=args.pruning_submass,
            jsonl=args.jsonl,
        )
    finally:
        if out is not sys.stdout:
            out.close()

    return 0


if __name__ == "__main__":
    exit(main())
