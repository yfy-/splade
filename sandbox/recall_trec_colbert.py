#!/usr/bin/env python3

import json
import argparse
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trec_file")
    parser.add_argument("colbert_out")
    args = parser.parse_args()

    with open(args.trec_file) as tfs:
        trec_lines = tfs.readlines()

    splade_dict = defaultdict(set)
    for tl in trec_lines:
        qid, _, cid, _ = tl.split(maxsplit=3)
        splade_dict[qid].add(int(cid))

    with open(args.colbert_out) as cfs:
        colbert_dict = json.load(cfs)

    tot_recall = 0.0
    for qid, colb_ret in colbert_dict.items():
        # import pdb; pdb.set_trace()
        if qid not in splade_dict:
            print(f"Should never happen {qid}")
            exit(1)

        num_rel = len(colb_ret)
        ret_rel = len(set(colb_ret) & splade_dict[qid])
        recall = ret_rel / num_rel
        print(f"Recall for {qid}: {recall}")
        tot_recall += recall

    print(f"Average recall: {tot_recall / len(colbert_dict)}")
