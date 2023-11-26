#!/usr/bin/env python3

import argparse
import torch
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
from splade_utils import splade_enhance, load_collection
import logging
import numpy as np
from tqdm import tqdm

MODEL_NAME = "naver/splade_v2_max"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("collection")
    parser.add_argument("-b", "--batch-size", type=int, default=100000)
    args = parser.parse_args()
    collection = load_collection(args.collection)
    splade = Splade(MODEL_NAME)
    if torch.cuda.is_available():
        logging.info("Moving model to cuda.")
        splade = splade.to("cuda")

    splade.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    num_doc = len(collection)
    num_batches = np.ceil(num_doc / args.batch_size)
    docs = list(collection.values())
    np_texts = np.array(docs, dtype="object")
    batches = np.array_split(np_texts, num_batches)
    docids = list(collection.keys())
    docid_idx = 0
    with tqdm(num_batches, desc="Enhancing doc batch") as pbar:
        for batch in batches:
            logging.critical(f"Batch size: {len(batch)}")
            doc_splade = splade_enhance(
                batch, splade, tokenizer, reverse_voc
            )
            logging.critical(f"Enhanced batch size: {len(doc_splade)}")
            assert len(doc_splade) == len(batch)
            for ds in doc_splade:
                print(f"{docids[docid_idx]}\t{ds}")
                docid_idx += 1


        pbar.update(1)

    return 0


if __name__ == "__main__":
    exit(main())
