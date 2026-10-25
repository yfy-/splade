# SPLADE (Sparcent fork)

SPLADE extended to build the sparse centroid documents and queries of
[Sparcent](https://github.com/yfy-/sparcent).

## What this fork adds

- `splade/sparcent/centroid_docs.py`: encodes a collection with SPLADE
  (batches stored as sparse tensors and resumed if interrupted), then
  for every group of passages, e.g. the passages of a ColBERT centroid
  (`-d`, a padded tensor of passage ids per group), averages the
  vectors of the first `-n` ratio of passages, prunes them to an L1
  submass `-p`, quantizes them to 8 bits and writes terms repeated by
  weight (PISA input) or JSONL (`--jsonl`). Without `-d` every passage
  is its own document.

  ```bash
  python -m splade.sparcent.centroid_docs collection.tsv encodings/ \
      naver/splade-cocondenser-ensembledistil compose-cache/ \
      -d sample_pids_count.pt -n 0.01 -p 0.05 -o centroid_docs.tsv.gz
  ```
- `splade/sparcent/queries.py`: `encode_sparse_queries()` encodes
  queries into 8-bit term weights (as used by the ColBERT fork at
  search time), also as a script writing `qid: term term ...` or
  JSONL: `python -m splade.sparcent.queries
  naver/splade-cocondenser-ensembledistil queries.tsv > queries.pisa`
- LoRA fine-tuning (`lora_tuning: true` in the training config, `-l`
  in the scripts above; requires `peft`).
- `scripts/` holds one-off experiment scripts (not maintained).

## Acknowledgements

This is a fork of [SPLADE](https://github.com/naver/splade) by NAVER
LABS Europe; all credit for SPLADE goes to its authors. See the
original repository for training, evaluation and the pretrained
models, and cite [SPLADE](https://arxiv.org/abs/2107.05720)
(SIGIR'21), [SPLADE v2](https://arxiv.org/abs/2109.10086) and
[SPLADE++](https://arxiv.org/abs/2205.04733) (SIGIR'22).
