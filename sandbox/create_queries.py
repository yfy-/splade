#!/usr/bin/env python3

import argparse
import torch
import numpy as np
from typing import Tuple, List
from sys import stdin
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
import logging

from splade_utils import splade_encode, quantize_local_sparse_8bit, \
    recreate_text

MODEL_NAME = "naver/splade_v2_distil"

def load_queries(query_file_stream) -> Tuple[List[str], List[str]]:
    query_lines = query_file_stream.readlines()
    n = len(query_lines)
    qids = [None] * n
    queries = [None] * n
    for i, ql in enumerate(query_lines):
        qids[i], queries[i] = ql.rstrip().split("\t")

    return qids, queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "queries", nargs="?", default=stdin, type=argparse.FileType()
    )
    logging.basicConfig(level=logging.DEBUG)
    parser.add_argument("-b", "--batch-size", type=int, default=100000)
    args = parser.parse_args()
    qids, queries = load_queries(args.queries)
    queries = np.array(queries, dtype="object")
    splade = Splade(MODEL_NAME)
    if torch.cuda.is_available():
        logging.info("Moving model to cuda")
        splade = splade.to("cuda")
    splade.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    num_batches = np.ceil(len(queries) / args.batch_size)
    batches = np.array_split(queries, num_batches)
    query_idx = 0
    for batch in batches:
        sparse_queries = splade_encode(
            batch, splade, tokenizer, is_query=True
        )
        # For query quantization we do not need to keep a global scale
        quantized = quantize_local_sparse_8bit(sparse_queries)
        reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
        enhanced_queries = recreate_text(quantized, reverse_voc)
        for eq in enhanced_queries:
            print(f"{qids[query_idx]}: {eq}")
            query_idx += 1

    return 0

if __name__ == "__main__":
    exit(main())
