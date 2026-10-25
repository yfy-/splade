#!/usr/bin/env python3

"""Encode queries with SPLADE in PISA (terms repeated by weight) or JSONL format."""

import argparse
import json
import logging
from pathlib import Path
from sys import stdin
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from splade.models.transformer_rep import Splade
from splade.sparcent.utils import (
    load_tokenizer_and_model,
    quantize_local_sparse_8bit,
    recreate_text,
    splade_encode,
)


def load_queries(query_file_stream) -> Tuple[List[str], List[str]]:
    query_lines = query_file_stream.readlines()
    n = len(query_lines)
    qids = [None] * n
    queries = [None] * n
    for i, ql in enumerate(query_lines):
        qids[i], queries[i] = ql.rstrip().split("\t")

    return qids, queries


def encode_sparse_queries(
    queries: List[str],
    splade: Splade,
    tokenizer,
    batch_size=100000,
    quantize=True,
    device: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Encode queries into term weights.

    With `quantize`, the weights of each query are quantized to 8 bits.
    """
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
    queries = np.array(queries, dtype="object")
    num_batches = np.ceil(len(queries) / batch_size)
    sparse_queries = []
    for batch in np.array_split(queries, num_batches):
        encoded = splade_encode(
            batch, splade, tokenizer, is_query=True, device=device
        )
        if quantize:
            encoded = quantize_local_sparse_8bit(encoded)

        sparse_queries.extend(recreate_text(encoded, reverse_voc))

    return sparse_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name_or_dir")
    parser.add_argument(
        "queries", nargs="?", default=stdin, type=argparse.FileType()
    )
    logging.basicConfig(level=logging.DEBUG)
    parser.add_argument("-b", "--batch-size", type=int, default=100000)
    parser.add_argument("-l", "--lora-model-dir")
    parser.add_argument("-j", "--jsonl", action="store_true")
    args = parser.parse_args()
    qids, queries = load_queries(args.queries)
    lora_model_dir = args.lora_model_dir
    if lora_model_dir is not None:
        lora_model_dir = Path(lora_model_dir)
    tokenizer, splade = load_tokenizer_and_model(
        args.model_name_or_dir, lora_model_dir
    )
    if torch.cuda.is_available():
        logging.info("Moving model to cuda")
        splade = splade.to("cuda")

    splade.eval()
    sparse_queries = encode_sparse_queries(
        queries,
        splade,
        tokenizer,
        batch_size=args.batch_size,
        quantize=not args.jsonl,
    )
    for qid, query, sq in zip(qids, queries, sparse_queries):
        if args.jsonl:
            print(json.dumps({"id": qid, "content": query, "vector": sq}))
        else:
            print(f"{qid}: ", end="")
            for q_term, term_w in sq.items():
                for _ in range(term_w):
                    print(f"{q_term} ", end="")

            print("")

    return 0


if __name__ == "__main__":
    exit(main())
