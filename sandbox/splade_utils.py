"""Generic utilities for the SPLADE model."""

from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
import torch
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from functools import partial
import logging
from pathlib import Path
from peft.mapping import get_peft_model
from peft.tuners.lora.config import LoraConfig


def load_collection(collection_path: str) -> Dict[int, str]:
    with open(collection_path) as cfs:
        collection_lines = cfs.readlines()

    docs = {}
    for cl in collection_lines:
        docid, doc = cl.rstrip().split("\t")
        docs[int(docid)] = doc

    return docs


def load_tokenizer_and_model(
    model_name_or_dir: str, lora_model_dir: Optional[Path]
) -> Tuple[AutoTokenizer, Splade]:
    """Load a splade tokenizer and model."""
    tok_model_name = model_name_or_dir
    splade = Splade(model_name_or_dir)
    if lora_model_dir is not None:
        tok_model_name = lora_model_dir
        peft_config = LoraConfig(target_modules=["q_lin", "k_lin"])
        splade = get_peft_model(splade, peft_config)
        model_dict = torch.load(lora_model_dir / "model.tar")
        splade.load_state_dict(model_dict["model_state_dict"])

    return AutoTokenizer.from_pretrained(tok_model_name), splade


def splade_encode(
    texts: List[str],
    splade: Splade,
    tokenizer: AutoTokenizer,
    batch_size=16,
    is_query=False,
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
                try:
                    tokens = tokenizer(
                        list(batch),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    if torch.cuda.is_available():
                        tokens = tokens.to("cuda")

                    text_rep = splade(**{splade_kwargs: tokens})[
                        splade_out
                    ].cpu()
                    next_processed = curr_processed + len(batch)
                    sparse_docs[curr_processed:next_processed] = text_rep
                    curr_processed = next_processed
                except Exception as ex:
                    logging.error(batch)
                    raise ex
                pbar.update(len(batch))

    return sparse_docs


def quantize_sparse_8bit(
    sparse_text: torch.tensor, glob_max: float = None
) -> torch.tensor:
    if glob_max is None:
        glob_max = torch.max(sparse_text)

    scale = glob_max / 256
    return torch.quantize_per_tensor(
        sparse_text, scale, 0, torch.quint8
    ).int_repr()


def quantize_local_sparse_8bit(sparse_text: torch.tensor) -> torch.tensor:
    scale = torch.max(sparse_text, axis=1).values / 256
    quantized = torch.quantize_per_tensor(
        list(sparse_text),
        scale,
        torch.zeros_like(scale, dtype=int),
        torch.quint8,
    )
    return torch.stack([quant.int_repr() for quant in quantized])


def recreate_text(
    quantized: torch.tensor, reverse_voc: Dict[int, str]
) -> List[Dict[str, int]]:
    n = quantized.shape[0]
    vocab_indices = quantized.nonzero()
    logging.debug(f"Num non-zero: {len(vocab_indices)}")
    enhanced_docs = [dict() for _ in range(n)]
    prev_i = 0
    with tqdm(total=n, desc="Text recreate") as pbar:
        for doc_i, term_i in vocab_indices:
            doc_i = doc_i.item()
            term_i = term_i.item()
            enhanced_docs[doc_i][reverse_voc[term_i]] = quantized[doc_i, term_i]
            if doc_i != prev_i:
                pbar.update(doc_i - prev_i)
                prev_i = doc_i

    return enhanced_docs


def splade_enhance(
    docs: List[str],
    splade: Splade,
    tokenizer: AutoTokenizer,
    reverse_voc: Dict[int, str],
    batch_size=16,
    jobs=1,
) -> List[str]:
    sparse_docs = splade_encode(docs, splade, tokenizer, batch_size=batch_size)
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
