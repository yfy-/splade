"""Generic utilities for the SPLADE model."""

from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
import torch
from typing import List, Dict
from tqdm import tqdm
import numpy as np
from functools import partial


def load_collection(collection_path: str) -> Dict[int, str]:
    with open(collection_path) as cfs:
        collection_lines = cfs.readlines()

    docs = {}
    for cl in collection_lines:
        docid, doc = cl.rstrip().split("\t")
        docs[int(docid)] = doc

    return docs


def splade_encode(
        texts: List[str], splade: Splade, tokenizer: AutoTokenizer,
        batch_size=16, is_query=False
) -> torch.Tensor:
    n = len(texts)
    splade_kwargs = "d_kwargs"
    splade_out = "d_rep"
    if is_query:
        splade_kwargs = "q_kwargs"
        splade_out = "q_rep"

    with torch.no_grad():
        np_texts = np.array(texts, dtype="object")
        batches = np.array_split(np_texts, np.ceil(n / batch_size))
        curr_processed = 0
        sparse_docs = torch.zeros((n, splade.output_dim))

        with tqdm(total=n, desc="Splade encode") as pbar:
            for batch in batches:
                tokens = tokenizer(
                    list(batch), return_tensors="pt", padding=True
                )
                if torch.cuda.is_available():
                    tokens = tokens.to("cuda")

                text_rep = splade(**{splade_kwargs: tokens})[splade_out].cpu()
                next_processed = curr_processed + len(batch)
                sparse_docs[curr_processed: next_processed] = text_rep
                curr_processed = next_processed
                pbar.update(len(batch))

    return sparse_docs


def quantize_sparse_8bit(sparse_text: torch.tensor) -> torch.tensor:
    scale = torch.max(sparse_text) / 256
    return torch.quantize_per_tensor(
        sparse_text, scale, 0, torch.quint8
    ).int_repr()


def recreate_text(
        quantized: torch.tensor, reverse_voc: Dict[int, str]
) -> List[str]:
    n = quantized.shape[0]
    vocab_indices = quantized.nonzero()
    enhanced_docs = [""] * n
    prev_i = 0
    with tqdm(total=n, desc="Text recreate") as pbar:
        doc_toks = []
        for doc_i, term_i in vocab_indices:
            doc_i = doc_i.item()
            term_i = term_i.item()
            if doc_i != prev_i:
                enhanced_docs[prev_i] = " ".join(doc_toks)
                doc_toks.clear()
                pbar.update(1)
                prev_i = doc_i

            doc_toks.append(
                " ".join([reverse_voc[term_i]] * quantized[doc_i, term_i])
            )
        enhanced_docs[prev_i] = " ".join(doc_toks)
        pbar.update(1)

    return enhanced_docs


def splade_enhance(
        docs: List[str], splade: Splade, tokenizer: AutoTokenizer,
        reverse_voc: Dict[int, str], batch_size=16, jobs=1
) -> List[str]:
    sparse_docs = splade_encode(docs, splade, tokenizer)
    quantized = quantize_sparse_8bit(sparse_docs)
    del sparse_docs
    quantized_batches = np.array_split(quantized, jobs)
    recreate_text_partial = partial(recreate_text, reverse_voc=reverse_voc)
    with torch.multiprocessing.get_context("spawn").Pool(jobs) as pool:
        recreated_batches = pool.map(recreate_text_partial, quantized_batches)

    flattened = []
    for batch in recreated_batches:
        for d in batch:
            flattened.append(d)

    return flattened
