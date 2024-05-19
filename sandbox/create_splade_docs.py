#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from functools import partial

import numpy as np
from peft.mapping import get_peft_model
from peft.tuners.lora.config import LoraConfig
import torch
from transformers import AutoTokenizer

from splade.models.transformer_rep import Splade
from splade_utils import (
    splade_encode,
    load_collection,
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
    parser.add_argument("-l", "--lora-model-name-or-dir")
    args = parser.parse_args()
    tok_model_name = args.model_name_or_dir
    if args.lora_model_name_or_dir:
        tok_model_name = args.lora_model_name_or_dir
    tokenizer = AutoTokenizer.from_pretrained(tok_model_name)
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
        splade = Splade(args.model_name_or_dir)
        if args.lora_model_name_or_dir:
            peft_config = LoraConfig(target_modules=["q_lin", "k_lin"])
            splade = get_peft_model(splade, peft_config)
            model_dict = torch.load(
                Path(args.lora_model_name_or_dir) / "model.tar"
            )
            splade.load_state_dict(model_dict["model_state_dict"])

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

    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    docids = list(collection.keys())
    docid_idx = 0
    batch_idx = 0
    global_max_weight = torch.load(max_weight_path)
    while True:
        logging.info(f"Loading: {batch_idx}")
        batch_path = work_dir / f"splade_encode_{batch_idx}.npz"
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
            with torch.multiprocessing.get_context("spawn").Pool(
                args.jobs
            ) as pool:
                recreated = pool.map(recreate_text_partial, quantized_batches)
        else:
            recreated = [recreate_text_partial(qb) for qb in quantized_batches]

        for qb in recreated:
            for d in qb:
                print(f"{docids[docid_idx]}\t{d}")
                docid_idx += 1

        batch_idx += 1

    return 0


if __name__ == "__main__":
    exit(main())
