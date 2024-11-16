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
import copy
import datetime
import omegaconf
import os
import uuid
from urllib.request import DataHandler
from aotc import types


def create_workload_config(
    workload_info: types.WorkloadInfo,
    artifact_config: types.ArtifactConfig,
    name: str,
    date: datetime.datetime,
) -> types.WorkloadConfig:
  # Populate RunInfo
  # run_name = "megatron-lm-gpt-llama-30b-ubs2-tp4-pp2"
  run_name = f"{workload_info.framework}-{workload_info.model}"
  run_timestamp = datetime.datetime.now(tz=None).strftime("%Y-%m-%d_%H%M%S")
  user = os.environ.get("USER")
  run_uid = f"{run_name}-{run_timestamp}-{uuid.uuid4()}"
  run_info = types.RunInfo(
      uid=run_uid,
      experiment_id=name,
      experiment_date=date,
      user=user,
  )

  # Update artifact config with defaults
  artifact_config = copy.deepcopy(artifact_config)
  if not artifact_config.gcs_dir:
    artifact_config.gcs_dir = f"{artifact_config.gcs_root}/{name}/{run_uid}"
  if not artifact_config.local_dir:
    artifact_config.local_dir = f"{artifact_config.local_root}/{name}/{run_uid}"

  if not artifact_config.tensorboard_dir:
    artifact_config.tensorboard_dir = f"{artifact_config.gcs_dir}/tensorboard"

  return types.WorkloadConfig(
      run=run_info,
      workload=copy.deepcopy(workload_info),
      artifact_config=artifact_config,
  )

def profile_ranks_staggered(num_nodes: int, num_all: int = 2):
  gpus_per_node = 8
  ranks = []
  for node_idx in range(num_nodes):
    offset = node_idx * gpus_per_node
    if node_idx < num_all:
      ranks += [i + offset for i in range(gpus_per_node)]
    else:
      ranks += [offset]
  return ranks


def get_output_config(config_path: str) -> types.OutputConfig:
  cfg = omegaconf.OmegaConf.load(config_path)
  schema = omegaconf.OmegaConf.structured(types.OutputConfig)
  cfg = omegaconf.OmegaConf.merge(schema, cfg)
  return omegaconf.OmegaConf.to_object(cfg)


def experiment_config(
    name: str, output_config_path: str, skip_existing: bool = True
):
  return types.ExperimentConfig(
      name=name,
      output=get_output_config(output_config_path),
      skip_existing_results=skip_existing,
  )

def get_workload_info(
    framework: str,
    model: str,
    num_steps: int,
    warmup_steps: int,
    profiler_steps: int,
    timeout: int,
    profiler_rank: list[int] = []
):
  """ Helper function to create a workload config. """

  if profiler_steps == 0:
    profiling = None
  else:
    profiling = types.ProfilingConfig(
      start_step=num_steps - profiler_steps - 1,
      end_step=num_steps - 1,
      ranks=profiler_rank,
    )

  return types.WorkloadInfo(
      framework=framework,
      model=model,
      num_steps=num_steps,
      timeout=timeout,
      tuning_params={},
      runtime_env={},
      profiling=profiling,
      metrics_config=types.MetricsConfig(
          # Exclude warmup and profiled steps from the summary stats
          stats_from_step=warmup_steps,
          stats_to_step=num_steps - profiler_steps - 1,
      ),
  )

