#!/usr/bin/env python3

import argparse
from typing import Dict

import torch

from splade_utils import load_collection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trec_torch_file",
        help="Torch file that has retrieved results for queries.",
    )
    parser.add_argument(
        "centroid_pids", help="Torch file that maps each centroid to its pid."
    )
    parser.add_argument("collection", help="Collection tsv file")
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        required=True,
        help="Only use top-k for validation set from 'trec_torch_file'",
    )
    args = parser.parse_args()
    trec_res: Dict[int, torch.IntTensor] = torch.load(args.trec_torch_file)
    topk_docs = [
        trec[: args.top_k].unsqueeze(0) for _, trec in trec_res.items()
    ]
    unique_docs = torch.concat(topk_docs, dim=0).unique()
    centroid_pids = torch.load(args.centroid_pids)[:, 0]
    collection = load_collection(args.collection)
    for ud in unique_docs:
        ud = ud.item()
        print(f"{ud}\t{collection[centroid_pids[ud].item()]}")


if __name__ == "__main__":
    exit(main())
