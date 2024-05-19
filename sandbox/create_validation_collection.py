#!/usr/bin/env python3

import argparse
import json
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
        "qrel_file",
        help="Query relevance judgement to add documents from it to the "
        "collection.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        required=True,
        help="Only use top-k for validation set from 'trec_torch_file'",
    )
    args = parser.parse_args()
    trec_res: Dict[int, torch.IntTensor] = torch.load(args.trec_torch_file)

    # Add documents retrieved from splade baseline
    docs = [
        trec[: args.top_k] for _, trec in trec_res.items()
    ]

    # Add documents retrieved from colbert ground truth
    with open(args.qrel_file) as qfs:
        qrels = json.load(qfs)

    qrel_docs = []
    for _, qrel_dict in qrels.items():
        for docid, _ in qrel_dict.items():
            qrel_docs.append(int(docid))

    docs.append(torch.IntTensor(qrel_docs))
    unique_docs = torch.concat(docs).unique()
    centroid_pids = torch.load(args.centroid_pids)[:, 0]
    collection = load_collection(args.collection)
    for ud in unique_docs:
        ud = ud.item()
        print(f"{ud}\t{collection[centroid_pids[ud].item()]}")


if __name__ == "__main__":
    exit(main())
