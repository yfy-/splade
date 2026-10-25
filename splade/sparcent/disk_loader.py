"""Row-batched dense tensors that are swapped to disk as sparse tensors."""

from __future__ import annotations

import logging
from pathlib import Path
from shutil import copytree, rmtree
from typing import Tuple

import numpy as np
import torch


class BatchSparseTensorDiskLoaderIterator:
    """Iterator for batched tensors serializable to disk."""

    def __init__(self, loader: BatchSparseTensorDiskLoader, immut=False):
        """Initialize the iterator.

        Args:
          loader: Batch tensor disk loader.
          immut: True if the batched tensors are used for read-only purposes.
        """
        self.loader = loader
        self.curr = 0
        self.immut = immut

    def __iter__(self):
        """Create an iterator."""
        return self

    def __next__(self) -> torch.Tensor:
        """Get next batch."""
        if self.curr >= self.loader.shape[0]:
            raise StopIteration

        res, _ = (
            self.loader.getitem_immut(self.curr)
            if self.immut
            else self.loader[self.curr]
        )
        self.curr += self.loader.batch_size
        return res


class BatchSparseTensorDiskLoader:
    def __init__(
        self, tensor_dir: Path, shape: Tuple[int, int], batch_size=int(1e6)
    ):
        self.tensor_dir = tensor_dir
        self.work_dir = tensor_dir / "workdir"
        self.shape = shape
        self.batch_size = min(batch_size, self.shape[0])
        self.num_batch = int(np.ceil(self.shape[0] / self.batch_size))
        self.tensor = None
        self.loaded_batch_idx = None

    def _save_batch(self):
        if self.loaded_batch_idx is None:
            return

        old_batch_path = self.work_dir / f"tensor_{self.loaded_batch_idx}.pt"
        logging.info(f"Saving {old_batch_path}")
        torch.save(self.tensor.to_sparse(), old_batch_path)

    def checkpoint(self, checkpoint: str):
        logging.info(f"Creating checkpoint: {checkpoint}")
        self._save_batch()
        copytree(self.work_dir, self.tensor_dir / checkpoint)

    def load_from_checkpoint(self, checkpoint: str) -> bool:
        checkpoint_p = self.tensor_dir / checkpoint
        if not checkpoint_p.exists():
            return False

        logging.info(f"Loading from checkpoint: {checkpoint}")
        copytree(checkpoint_p, self.work_dir, dirs_exist_ok=True)
        return True

    def __getitem__(self, row_idx: int) -> Tuple[torch.Tensor, int]:
        return self.__get_item(row_idx, save=True)

    def __get_item(self, row_idx: int, save=True):
        if row_idx >= self.shape[0]:
            raise KeyError(row_idx)

        batch_idx = row_idx // self.batch_size
        batch_tensor_idx = row_idx % self.batch_size
        if batch_idx != self.loaded_batch_idx:
            if save:
                self._save_batch()

            batch_path = self.work_dir / f"tensor_{batch_idx}.pt"
            if batch_path.exists():
                logging.info(f"Loading {batch_path}")
                self.tensor = torch.load(batch_path).to_dense()
            else:
                batch_size = self.batch_size
                if batch_idx == self.num_batch - 1:
                    batch_size = self.shape[0] - (
                        (self.num_batch - 1) * self.batch_size
                    )

                self.tensor = torch.zeros((batch_size, self.shape[1]))

            self.loaded_batch_idx = batch_idx

        return self.tensor, batch_tensor_idx

    def getitem_immut(self, row_idx):
        return self.__get_item(row_idx, save=False)

    def __enter__(self):
        self.work_dir.mkdir(exist_ok=True, parents=True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._save_batch()
        rmtree(self.work_dir)

    def batch_iter(self, immut=False) -> BatchSparseTensorDiskLoaderIterator:
        return BatchSparseTensorDiskLoaderIterator(self, immut=immut)
