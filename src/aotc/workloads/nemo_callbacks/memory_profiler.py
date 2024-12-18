# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Provides memory profiling for NeMo workloads until NeMo supports it natively."""

import os

from nemo.lightning import io
from nemo.utils import logging
from nemo.utils.get_rank import get_rank
from pytorch_lightning.callbacks.callback import Callback
import torch
from torch.utils.viz._cycles import warn_tensor_cycles


class MemoryProfileCallback(Callback, io.IOMixin):

  def __init__(self, dir: str = "/mem_profile", warn_cycles=True, ranks=[]):

    self.dir = dir
    self.ranks = self._process_ranks(ranks)

    os.makedirs(self.dir, exist_ok=True)
    logging.info(f"Torch memory profiles will be written to: {self.dir}")

    if warn_cycles:
      # Enable reference counter to detect and break reference cycles.
      # This helps prevent GPU memory leaks caused by circular references
      # between tensors.
      logging.info("Enabling reference cycle detector")
      warn_tensor_cycles()

  def _process_ranks(self, ranks):
    result = set()
    for rank in ranks:
      if isinstance(rank, str) and '-' in rank:
        try:
          start, end = map(int, rank.split('-'))
          if start < 0 or end < 0 or end < start:
            raise ValueError(f"Invalid range: {rank}")
          result.update(range(start, end+1))
        except ValueError as e:
          raise e
      else:
        try:
          result.add(int(rank))
        except ValueError:
          raise ValueError(f"Invalid int format: {rank}")

    return sorted(list(result))

  def enable_on_rank(self) -> bool:
    if not self.ranks:
      return True
    return get_rank() in self.ranks

  def setup(self, trainer, pl_module, stage) -> None:
    if torch.distributed.is_initialized() and self.enable_on_rank():
      torch.cuda.memory._record_memory_history(max_entries=100000)

  def on_train_end(self, trainer, pl_module) -> None:
    logging.info(
        f"on_train_batch_end rank: {get_rank()} mem:"
        f" {torch.cuda.memory_allocated()/1024/1024/1024} /"
        f" {torch.cuda.max_memory_reserved()/1024/1024/1024}"
    )

    if torch.distributed.is_initialized() and self.enable_on_rank():
      rank = get_rank()
      _snapshot_path = f"{self.dir}/memory_snapshot-rank{rank}.pickle"
      logging.info(f"Writing memory profile snapshot to {_snapshot_path}")
      torch.cuda.memory._dump_snapshot(f"{_snapshot_path}")
      torch.cuda.memory._record_memory_history(enabled=None)
      logging.info(
          f"Finished writing memory profile snapshot: {_snapshot_path}"
      )
