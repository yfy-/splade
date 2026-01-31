#!/usr/bin/env python3

"""Create splade docs in text format."""

from __future__ import annotations
import argparse
from collections import defaultdict
from functools import partial
import logging
from pathlib import Path
from psutil import virtual_memory
from shutil import rmtree, copytree
from typing import Tuple
import json

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


class BatchSparseTensorDiskLoaderIterator:
    """Iterator for batched tensors serializable to disk."""

    def __init__(self, loader: BatchSparseTensorDiskLoader, immut=False):
        """Initialize the iterator.

        Args:
          loader: Batch tensor disk loader.
          immut: True if the batched tensors are used for read-only purposes.
        """
        self.loader = loader
        self.curr = 0
        self.immut = immut

    def __iter__(self):
        """Create an iterator."""
        return self

    def __next__(self) -> torch.Tensor:
        """Get next batch."""
        if self.curr >= self.loader.shape[0]:
            raise StopIteration

        res, _ = (
            self.loader.getitem_immut(self.curr)
            if self.immut
            else self.loader[self.curr]
        )
        self.curr += self.loader.batch_size
        return res


class BatchSparseTensorDiskLoader:
    def __init__(
        self, tensor_dir: Path, shape: Tuple[int, int], batch_size=int(1e6)
    ):
        self.tensor_dir = tensor_dir
        self.work_dir = tensor_dir / "workdir"
        self.shape = shape
        self.batch_size = min(batch_size, self.shape[0])
        self.num_batch = int(np.ceil(self.shape[0] / self.batch_size))
        self.tensor = None
        self.loaded_batch_idx = None

    def _save_batch(self):
        if self.loaded_batch_idx is None:
            return

        old_batch_path = self.work_dir / f"tensor_{self.loaded_batch_idx}.pt"
        logging.info(f"Saving {old_batch_path}")
        torch.save(self.tensor.to_sparse(), old_batch_path)

    def checkpoint(self, checkpoint: str):
        logging.info(f"Creating checkpoint: {checkpoint}")
        self._save_batch()
        copytree(self.work_dir, self.tensor_dir / checkpoint)

    def load_from_checkpoint(self, checkpoint: str) -> bool:
        checkpoint_p = self.tensor_dir / checkpoint
        if not checkpoint_p.exists():
            return False

        logging.info(f"Loading from checkpoint: {checkpoint}")
        copytree(checkpoint_p, self.work_dir, dirs_exist_ok=True)
        return True

    def __getitem__(self, row_idx: int) -> Tuple[torch.Tensor, int]:
        return self.__get_item(row_idx, save=True)

    def __get_item(self, row_idx: int, save=True):
        if row_idx >= self.shape[0]:
            raise KeyError(row_idx)

        batch_idx = row_idx // self.batch_size
        batch_tensor_idx = row_idx % self.batch_size
        if batch_idx != self.loaded_batch_idx:
            if save:
                self._save_batch()

            batch_path = self.work_dir / f"tensor_{batch_idx}.pt"
            if batch_path.exists():
                logging.info(f"Loading {batch_path}")
                self.tensor = torch.load(batch_path).to_dense()
            else:
                batch_size = self.batch_size
                if batch_idx == self.num_batch - 1:
                    batch_size = self.shape[0] - (
                        (self.num_batch - 1) * self.batch_size
                    )

                self.tensor = torch.zeros((batch_size, self.shape[1]))

            self.loaded_batch_idx = batch_idx

        return self.tensor, batch_tensor_idx

    def getitem_immut(self, row_idx):
        return self.__get_item(row_idx, save=False)

    def __enter__(self):
        self.work_dir.mkdir(exist_ok=True, parents=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._save_batch()
        rmtree(self.work_dir)

    def batch_iter(self, immut=False) -> BatchSparseTensorDiskLoaderIterator:
        return BatchSparseTensorDiskLoaderIterator(self, immut=immut)


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
    parser.add_argument("--jsonl", action="store_true")
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

    # Take average of splade encoding for each centroid by loading
    # them batch by batch.
    compose_cache_path = Path(args.compose_cache_path)
    # use_cache = compose_cache_path.exists()
    mem_limit = virtual_memory().total // 4
    composed_shape = (len(docid2docs), len(tokenizer.vocab))
    batch_size = mem_limit // (composed_shape[1] * 4)
    logging.info(f"Composed batch_size={batch_size}")
    compose_cp = "compose-checkpoint"
    with BatchSparseTensorDiskLoader(
        compose_cache_path,
        composed_shape,
        batch_size=batch_size,
    ) as composed_docs_loader:
        if not composed_docs_loader.load_from_checkpoint(compose_cp):
            # Take nearest percent of passages for each centroid
            if args.nearest_percent < 1.0:
                psg_counts = torch.sum(docid2docs != -1, dim=1, keepdim=True)
                psg_counts_keep = torch.ceil(
                    psg_counts * args.nearest_percent
                ).int()
                remove_mask = (
                    torch.arange(docid2docs.shape[1]).repeat(
                        docid2docs.shape[0], 1
                    )
                    >= psg_counts_keep
                )
                docid2docs[remove_mask] = -1.0

            # Create a lookup dict where keys are passages and values are a
            # list of centroids they appear. This way we can load splade
            # vector batches in sequential order to not blow up the memory.
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
                    splade_encode_batch = load_batch(
                        work_dir, last_loaded_batch_idx
                    )
                    encode_batch_idx = i - total_proc

                for did in rlookup[i]:
                    composed_docs, cdi = composed_docs_loader[did]
                    composed_docs[cdi] += splade_encode_batch[
                        encode_batch_idx
                    ].clone()
                    num_psg_cent[did] += 1

            del rlookup
            c_batch_size = composed_docs_loader.batch_size
            i = 0
            for composed_docs in composed_docs_loader.batch_iter():
                composed_docs /= num_psg_cent[i : i + c_batch_size]
                i += c_batch_size

            composed_docs_loader.checkpoint(compose_cp)
        else:
            del docid2docs

        # Prune each centroid vector to its specified L1 submass.
        if args.pruning_submass < 1.0:
            logging.info(f"Pruning submass factor: {args.pruning_submass}")
            tot_set = 0
            for i in tqdm(
                range(composed_docs_loader.shape[0]),
                total=composed_docs_loader.shape[0],
                desc="Pruning",
            ):
                composed_docs, cdi = composed_docs_loader[i]
                doc_sorted, sort_idx = torch.sort(
                    composed_docs[cdi], descending=True
                )
                doc_cumsum = torch.cumsum(doc_sorted, dim=0)
                # Normalize with largest num
                doc_cumsum /= doc_cumsum[-1:].clone()
                search_idx = torch.searchsorted(
                    doc_cumsum, args.pruning_submass
                )
                keep_mask = torch.arange(len(sort_idx)) <= search_idx

                prune_mask = torch.ones(len(sort_idx), dtype=torch.bool)

                # Set most important dimensions to False so that
                # we do not set them to 0.0.
                prune_mask[sort_idx[keep_mask]] = False
                composed_docs[cdi][prune_mask] = 0.0
                tot_set += torch.sum(composed_docs[cdi] > 0.0).item()

            logging.info(f"Avg dim set: {tot_set / len(composed_docs)}")

        # Quantize and then reconstruct textual documents.
        reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
        if not args.jsonl:
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
                    quantized_batches = np.array_split(q_split, args.jobs)
                    recreate_text_partial = partial(
                        recreate_text, reverse_voc=reverse_voc
                    )
                    if args.jobs > 1:
                        with torch.multiprocessing.get_context("spawn").Pool(
                            args.jobs
                        ) as pool:
                            recreated = pool.map(
                                recreate_text_partial, quantized_batches
                            )
                    else:
                        recreated = [
                            recreate_text_partial(qb)
                            for qb in quantized_batches
                        ]

                    for qb in recreated:
                        for d in qb:
                            print(f"{docids[docidx]}\t", end="")
                            for term, weight in d.items():
                                for _ in range(weight):
                                    print(f"{term} ", end="")
                            print("")
                            docidx += 1
        else:
            docidx = 0
            for composed_docs in composed_docs_loader.batch_iter(immut=True):
                for q_split in composed_docs.split(5000):
                    split_batches = np.array_split(q_split, args.jobs)
                    recreate_text_partial = partial(
                        recreate_text, reverse_voc=reverse_voc
                    )
                    if args.jobs > 1:
                        with torch.multiprocessing.get_context("spawn").Pool(
                            args.jobs
                        ) as pool:
                            recreated = pool.map(
                                recreate_text_partial, split_batches
                            )
                    else:
                        recreated = [
                            recreate_text_partial(qb) for qb in split_batches
                        ]

                    for qb in recreated:
                        for d in qb:
                            jsonl_dict = {
                                "id": docids[docidx],
                                "content": collection[docids[docidx]],
                                "vector": d,
                            }
                            print(json.dumps(jsonl_dict))
    return 0


if __name__ == "__main__":
    exit(main())
