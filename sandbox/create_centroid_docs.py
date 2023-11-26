#!/usr/bin/env python3

import argparse
import torch
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
from splade_utils import splade_enhance, load_collection
import logging

MODEL_NAME = "naver/splade_v2_max"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("collection")
    parser.add_argument("sample_pids")
    args = parser.parse_args()

    sample_pids = torch.load(args.sample_pids)
    collection = load_collection(args.collection)
    pids_unq = sample_pids.flatten().unique()
    logging.info(f"pids_unq size pre: {len(pids_unq)}")
    pids_unq = pids_unq[pids_unq != -1]
    logging.info(f"pids_unq size post: {len(pids_unq)}")
    docs = [collection[p.item()] for p in pids_unq]

    splade = Splade(MODEL_NAME)
    if torch.cuda.is_available():
        logging.info("Moving model to cuda")
        splade = splade.to("cuda")

    splade.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    doc_splade = splade_enhance(docs, splade, tokenizer, reverse_voc)
    pid_index = {p.item(): i for i, p in enumerate(pids_unq)}
    for i, cpids in enumerate(sample_pids):
        cent_doc = " ".join(
            [
                doc_splade[pid_index[cp.item()]].rstrip() for cp in cpids
                if cp.item() != -1
            ]
        )

        print(f"{i}\t{cent_doc}")

    return 0

if __name__ == "__main__":
    exit(main())
