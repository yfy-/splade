#!/usr/bin/env python3

import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trec_file")
    parser.add_argument("num_query", type=int)
    parser.add_argument("out_path")
    args = parser.parse_args()

    with open(args.trec_file) as tfs:
        trec_lines = tfs.readlines()

    k = len(trec_lines) // args.num_query
    tensor = torch.full((args.num_query, k), -1, dtype=int)
    row = 0
    col = 0
    for tl in trec_lines:
        _, _, cid, _ = tl.split(maxsplit=3)
        tensor[row, col] = int(cid)
        col += 1
        if col == k:
            col = 0
            row += 1

    torch.save(tensor, args.out_path)
