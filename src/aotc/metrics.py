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
import dataclasses
import numpy as np
from typing import Any, List, Optional


from aotc import utils


@dataclasses.dataclass
class MetricMean:
  mean: float

  @classmethod
  def from_np(cls, values: np.ndarray):
    return cls(mean=float(np.mean(values)))

@dataclasses.dataclass
class MetricSimpleStats:
  mean: float
  min: float
  max: float
  num: int

  @classmethod
  def from_np(cls, values: np.ndarray):
    return cls(
        mean=float(np.mean(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        num=int(len(values)),
    )


@dataclasses.dataclass
class MetricIterStats:
  steps: List[int]
  scores: List[float]
  mean: float
  max: float
  min: float
  num: int

  @classmethod
  def from_np(cls, steps: np.ndarray, values: np.ndarray):
    return cls(
        steps=list(steps),  # Convert steps to list
        scores=list(values),  # Convert values to list
        mean=float(np.mean(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        num=int(len(values)),
    )


@dataclasses.dataclass
class MetricStats(MetricSimpleStats):
  std: float
  p50: float
  p90: float

  @classmethod
  def from_np(cls, values: np.ndarray):
    return cls(
        mean=float(np.mean(values)),
        min=float(np.min(values)),
        max=float(np.max(values)),
        num=int(len(values)),
        std=float(np.std(values)),
        p50=float(np.percentile(values, 50)),
        p90=float(np.percentile(values, 90)),
    )


@dataclasses.dataclass
class MetricPercentiles(MetricStats):
  p95: float
  p99: float


@dataclasses.dataclass
class WorkloadMetrics(utils.Unionable):
  """Metrics collected and returned by each workload implementation"""

  # WorkloadTask duration
  task_time: Optional[MetricSimpleStats] = None

  # In case of workload or extraction failures, these may not be set
  num_iterations: Optional[int] = None
  warmup_iter: Optional[int] = 0
  global_batch_size: Optional[int] = None
  seq_length: Optional[int] = None
  precision: Optional[str] = None
  optimizer: Optional[str] = None
  iteration_time: Optional[MetricStats] = None
  # memory usage on the device (likely from one arbitrary device)
  mem_usage_bytes: Optional[MetricSimpleStats] = None
  tokens_per_sec: Optional[MetricMean] = None
  throughput: Optional[MetricStats] = None
  loss: Optional[MetricStats] = None
  samples_for_convergence: Optional[int] = None
  metrics_accuracy_1: Optional[MetricIterStats] = None
  metrics_accuracy_2: Optional[MetricIterStats] = None

def get_task_time_metrics(
    time_taken: List[Any],
) -> WorkloadMetrics:
  # Metrics for all types of tasks
  task_time = MetricSimpleStats.from_np(
      np.array(time_taken)
  )
  return WorkloadMetrics(task_time=task_time)
