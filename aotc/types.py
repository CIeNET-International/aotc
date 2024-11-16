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
from abc import ABC, abstractmethod
import collections
import copy
import dataclasses
import datetime
import time
import functools
import json
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class ModelParams:
  global_batch_size: int
  seq_length: int


class SystemTimeout(Exception):
  """System is not ready after timeout is met"""


@dataclasses.dataclass
class DistributedTaskInfo:
  """Rank info for a distributed task."""

  rank: int
  num_nodes: int
  world_size: int  # == num_nodes * num_tasks_per_node
  local_rank: int = 0
  num_tasks_per_node: int = 1

  @property
  def node_rank(self):
    return self.rank // self.num_tasks_per_node


@dataclasses.dataclass
class ArtifactConfig:
  # Write artifacts to this GCS directory.
  gcs_root: str

  # Flag to save artifacts to GCS
  gcs_enabled: bool = True

  # By default, artifacts are exported to $gcs_root/<experiment_id>/<run_id>
  # Set a directory here to overwrite the default.
  gcs_dir: Optional[str] = None
  # Set to overwrite the tensorboard directory.
  # By default, writes tensorboard files to $gcs_dir/tensorboard
  tensorboard_dir: Optional[str] = None

  # Used to collect artifacts locally during the run and then upload to GCS
  # after the experiment completes.
  # All files in this folder will be uploaded as-is to {gcs_dir} on each node
  # where files exist. Care should be taken to name files to be unique across
  # all nodes as to avoid name conflicts on GCS.
  local_root: Optional[str] = None
  # default: {local_root}/<experiment_id>/<run_id>
  local_dir: Optional[str] = "/artifacts"

  # Whether to collect tensorboard to /tmp ({local_dir}/tensorboard) during the
  # run. The tensorboard will be copied to GCS after the run completes.
  # This can improve performance, but if the workload fails/crashes prematurely,
  # the tensorboard output might be missing from GCS.
  tensorboard_tmp: bool = True
  # Rank of the node to collect tensorboard from.
  # Required if `tensorboard_tmp` is True.
  tensorboard_rank: Optional[int] = None

  # Extra directories to collect locally and write to GCS
  # maps from `artifact_name` to the local dir or file
  # If the value is a dir, it will be zipped before upload as
  # <artifact_name>.zip
  collect_dirs: Optional[Dict[str, str]] = None


@dataclasses.dataclass
class OutputConfig:
  artifact_config: ArtifactConfig

@dataclasses.dataclass
class ExperimentConfig:
  name: str
  output: OutputConfig
  skip_existing_results: bool = False


@dataclasses.dataclass
class WorkloadTaskResult:
  time_taken: float
  run_result: Dict[str, Any]


@dataclasses.dataclass
class RunInfo:
  uid: str
  experiment_id: str
  experiment_date: datetime.datetime
  user: Optional[str] = None


@dataclasses.dataclass
class WorkloadSummary:
  """Schema for WorkloadInfo, stringified dicts"""

  framework: str  # name of workload type (megatronlm, pytorch, pax)
  model: str  # name of model or benchmarked workload (gpt30b, allreduce)
  num_steps: int  # number of steps or iterations to run
  tuning_params: str
  runtime_env: Optional[str] = None
  profiling: Optional[str] = None


@dataclasses.dataclass
class HardwareSystemSummary:
  num_nodes: int
  accelerator_type: str  # H100, v5p, ...
  num_devices: Optional[int] = None
  node_type: Optional[str] = None  # optional, e.g. A3-debian
  provider: Optional[str] = (
      None  # optional, e.g. MigNodeProvider, QueuedResource, etc.
  )
  topology: Optional[str] = None  # e.g. 8x8x16


@dataclasses.dataclass
class SoftwareSystemSummary:
  vm_image: Optional[str] = None  # image version running this code
  docker_image: Optional[str] = None  # image version running this code
  os_type: Optional[str] = None  # e.g. "debian", "cos"
  version_vector: Optional[str] = None  # uid to all versions used


@dataclasses.dataclass
class SystemSummary:
  hardware: HardwareSystemSummary
  software: SoftwareSystemSummary



@dataclasses.dataclass
class ProfilingConfig:
  start_step: int
  end_step: Optional[int] = None
  ranks: List[int] = dataclasses.field(default_factory=lambda: [0])


@dataclasses.dataclass
class MetricsConfig:
  # stats are summarized in the range [stats_from_step, stats_to_step]
  # This defines a `slice(stats_from_step, stats_to_step)`.
  stats_from_step: Optional[int] = None
  stats_to_step: Optional[int] = None


@dataclasses.dataclass
class WorkloadInfo:
  framework: str  # name of workload type (megatronlm, pytorch, pax)
  model: str  # name of model or benchmarked workload (gpt30b, allreduce)
  num_steps: int  # number of steps or iterations to run
  # timeout in seconds for the workload to complete.
  # The workload will be cancelled if it runs longer than this.
  timeout: int
  tuning_params: Dict[str, Any]
  # Whether to use mock data during training
  # this is useful for tuning sweeps to find the best tuning parameters.
  # Using real data can cause much higher variance in the results
  # and make it harder to find the best performing parameters.
  mock_data: bool = True
  # Extra ray runtime env to set environment vars
  runtime_env: Optional[Dict[str, Any]] = None
  profiling: Optional[ProfilingConfig] = None
  metrics_config: Optional[MetricsConfig] = None
  framework_version: Optional[str] = (
      None  # version of the framework, i.e. a branch name
  )

  def to_workload_summary(self) -> WorkloadSummary:
    # The schema for writing out to bigquery doesn't support dicts.
    # So we convert non-trivial data to json strings.
    return WorkloadSummary(
        framework=self.framework,
        model=self.model,
        num_steps=self.num_steps,
        tuning_params=json.dumps(self.tuning_params),
        runtime_env=json.dumps(self.runtime_env) if self.runtime_env else None,
        profiling=json.dumps(dataclasses.asdict(self.profiling))
        if self.profiling
        else None,
    )


@dataclasses.dataclass
class WorkloadConfig:
  run: RunInfo
  workload: WorkloadInfo
  artifact_config: ArtifactConfig

  def shallow_dict(self):
    """Returns a shallow dict of this data class.

    When initializing subclasses of this dataclass, this can be used to create a
    copy of the base class.

    Example:

      @dataclasses.dataclass
      class MyWorkloadConfig(WorkloadConfig):
        my_field: int

      config: WorkloadConfig = ...
      my_config = MyWorkloadConfig(**config.shallow_dict(), my_field=...)
    """
    return {
        field.name: copy.deepcopy(getattr(self, field.name))
        for field in dataclasses.fields(self)
    }

@functools.total_ordering
@dataclasses.dataclass
class GcePhysicalHost:
  cluster: str
  rack: str
  host: str

  def __lt__(self, other: "GcePhysicalHost") -> bool:
    return (
        self.cluster < other.cluster
        or (self.cluster == other.cluster and self.rack < other.rack)
        or (
            self.cluster == other.cluster
            and self.rack == other.rack
            and self.host < other.host
        )
    )

  @classmethod
  def from_string(cls, physical_host_str: str) -> "GcePhysicalHost":
    parts = physical_host_str.split("/")
    assert len(parts) == 4  # expect format `/cluster/rack/host`
    cluster, rack, host = parts[1:4]
    return GcePhysicalHost(cluster=cluster, rack=rack, host=host)

  @classmethod
  def from_metadata(
      cls, metadata: Dict[str, Any]
  ) -> Optional["GcePhysicalHost"]:
    physical_host: Optional[str] = (
        metadata["instance"].get("attributes", {}).get("physical_host")
    )
    if not physical_host:
      return None
    return GcePhysicalHost.from_string(physical_host_str=physical_host)


@dataclasses.dataclass
class NodeMetadata:
  """Node metadata, associated with each task result."""

  node_id: str
  hostname: str
  address: Optional[str] = None
  location: Optional[GcePhysicalHost] = None  # Assuming GcePhysicalHost is defined elsewhere
  # Index within a group (e.g. the executor). Set only when part of a SpdmResult.
  index: Optional[int] = None

  def remote_address(self) -> str:
    if self.address is None:
      return self.hostname
    else:
      return self.address


@dataclasses.dataclass
class TaskMetadata:
  """Metadata for a single task."""

  node: NodeMetadata
  distr: DistributedTaskInfo


class SystemErrorChecker:
    def is_system_error(self, error):
        raise NotImplementedError


class StandaloneErrorChecker(SystemErrorChecker):
  def is_system_error(self, error):
    if isinstance(error, SystemTimeout):
      return True
    return False


@dataclasses.dataclass
class TaskResult:
  metadata: TaskMetadata
  run_result: WorkloadTaskResult
  start_timestamp: float
  end_timestamp: float = 0
  exception: Optional[Exception] = None
  error_checker: SystemErrorChecker = StandaloneErrorChecker()

  @property
  def duration(self) -> float:
    return self.end_timestamp - self.start_timestamp

  def get_result(self):
    if self.exception:
      raise self.exception
    return self.result

  def has_error(self):
    return self.exception is not None

  def set_result(self, result):
    self.end_timestamp = time.time()
    self.result = result

  def set_error(self, error: Exception):
    self.end_timestamp = time.time()
    self.exception = error
