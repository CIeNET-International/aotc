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
import datetime
from typing import Optional


@dataclasses.dataclass
class WorkloadBenchmarkV2Schema:
  run_id: str

  # Unique model id to map model info table
  model_id: str

  # Foreign  key to join with  software info
  software_id: str
  # Foreign  key to join with hardware info
  hardware_id: str
  hardware_num_chips: int
  hardware_num_nodes: int

  result_success: bool

  configs_framework: str
  configs_env: str
  configs_container_version: str

  logs_artifact_directory: str

  update_person_ldap: str

  run_source: str = "manual"
  is_run_externally_visible: bool = False
  run_type: str = "perf_optimization"

  run_release_status: Optional[str] = "local"

  experiment_id: Optional[str] = None

  workload_gbs: Optional[int] = None
  workload_precision: Optional[str] = None
  workload_optimizer: Optional[str] = None
  workload_others: Optional[str] = None
  workload_manager: Optional[str] = None
  workload_type: str = "training"
  workload_sequence_length: Optional[int] = None

  metrics_step_time: Optional[float] = None
  metrics_mfu: Optional[float] = None
  metrics_tokens_per_second: Optional[float] = None
  metrics_e2e_time: Optional[float] = None
  metrics_steps_for_convergence: Optional[int] = None
  metrics_other: Optional[str] = None

  hardware_num_superblocks: Optional[str] = None
  hardware_num_slices: Optional[str] = None
  hardware_topology: Optional[str] = None
  hardware_num_cores: Optional[int] = None
  result_error: Optional[str] = None
  hardware_nccl_driver_nickname: Optional[str] = None

  configs_xla_flags: Optional[str] = None
  configs_dataset: Optional[str] = None
  configs_reviewer: Optional[str] = None
  configs_other: Optional[str] = None

  logs_profile: Optional[str] = None
  logs_cloud_logs: Optional[str] = None
  logs_comments: Optional[str] = None
  logs_other: Optional[str] = None

  reviewer_ldap: str = ""
  update_timestamp: datetime.datetime = datetime.datetime.now()
