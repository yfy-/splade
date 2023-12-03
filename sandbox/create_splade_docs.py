#!/usr/bin/env python3

import argparse
import torch
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
from splade_utils import (
    splade_encode,
    load_collection,
    quantize_sparse_8bit,
    recreate_text
)
import logging
import numpy as np
from pathlib import Path
from functools import partial

MODEL_NAME = "naver/splade_v2_max"


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("collection")
    parser.add_argument("work_dir")
    parser.add_argument("-b", "--batch-size", type=int, default=100000)
    parser.add_argument("-e", "--encode-batch-size", type=int, default=16)
    parser.add_argument("-j", "--jobs", type=int, default=1)
    parser.add_argument("-s", "--skip-encode", action="store_true")
    parser.add_argument("--encode-batch", type=int, default=0)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collection = load_collection(args.collection)
    logging.info(f"Collection size: {len(collection)}")
    work_dir = Path(args.work_dir)
    max_weight_path = work_dir / "global_max_weight.pt"
    if not args.skip_encode:
        global_max_weight = 0.0
        if max_weight_path.exists():
            global_max_weight = torch.load(max_weight_path)

        num_doc = len(collection)
        num_batches = np.ceil(num_doc / args.batch_size)
        docs = list(collection.values())
        np_texts = np.array(docs, dtype="object")
        batches = np.array_split(np_texts, num_batches)
        splade = Splade(MODEL_NAME)
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
            global_max_weight = max(
                global_max_weight, torch.max(batch_encoded)
            )
            as_np = batch_encoded.detach().cpu().resolve_conj().resolve_neg().numpy()
            np.savez_compressed(
                work_dir / f"splade_encode_{i}.npz",
                arr=as_np
            )
            torch.save(global_max_weight, max_weight_path)
            # torch.save(batch_encoded, work_dir / f"splade_encode_{i}.npy)

    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    docids = list(collection.keys())
    docid_idx = 0
    batch_idx = 0
    global_max_weight = torch.load(max_weight_path)
    while True:
        logging.info(f"Loading: {batch_idx}")
        batch_path = work_dir / f"splade_encode_{batch_idx}.npz"
        batch_out_path = work_dir / f"msmarco_splade_{batch_idx}.tsv"
        if not batch_path.exists():
            break
        batch_encoded = torch.Tensor(np.load(batch_path)["arr"])
        logging.info(f"Quantizing: {batch_idx}")
        quantized = quantize_sparse_8bit(
            batch_encoded, glob_max=global_max_weight
        )
        del batch_encoded
        quantized_batches = np.array_split(quantized, args.jobs)
        recreate_text_partial = partial(recreate_text, reverse_voc=reverse_voc)
        if args.jobs > 1:
            with torch.multiprocessing.get_context(
                    "spawn"
            ).Pool(args.jobs) as pool:
                recreated = pool.map(
                    recreate_text_partial, quantized_batches
                )
        else:
            recreated = [recreate_text_partial(qb)
                         for qb in quantized_batches]
        # recreated = recreate_text(quantized, reverse_voc)
        with batch_out_path.open("w") as bofs:
            for qb in recreated:
                for d in qb:
                    bofs.write(f"{docids[docid_idx]}\t{d}\n")
                    docid_idx += 1

        batch_idx += 1

    return 0


if __name__ == "__main__":
    exit(main())
