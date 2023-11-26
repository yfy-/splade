#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from typing import Tuple, List
from sys import stdin
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
import logging

from splade_utils import splade_encode, quantize_sparse_8bit, recreate_text

MODEL_NAME = "naver/splade_v2_max"

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
    args = parser.parse_args()
    qids, queries = load_queries(args.queries)
    splade = Splade(MODEL_NAME)
    if torch.cuda.is_available():
        logging.info("Moving model to cuda")
        splade = splade.to("cuda")
    splade.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sparse_queries = splade_encode(queries, splade, tokenizer, is_query=True)
    quantized = quantize_sparse_8bit(sparse_queries)
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    enhanced_queries = recreate_text(quantized, reverse_voc)
    for i, eq in enumerate(enhanced_queries):
        print(f"{qids[i]}: {eq}")

    return 0

if __name__ == "__main__":
    exit(main())
