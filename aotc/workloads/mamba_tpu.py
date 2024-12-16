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
import dataclasses
import json
import logging
import os
import pprint
import traceback
from typing import List, Optional
import numpy as np
from aotc import gpu_utils
from aotc import metric_extraction
from aotc import metrics as aotc_metrics
from aotc import types
from aotc import utils
from aotc import workload
import omegaconf
import yaml

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MambaWorkloadTask(workload.WorkloadTask):

  def __init__(self, config: types.WorkloadConfig):
    self._config: types.WorkloadConfig = config

  def _prep_node(
      self, distributed_task_info: types.DistributedTaskInfo
  ) -> None:
    pass

  def _run_workload(
      self, distributed_task_info: types.DistributedTaskInfo
  ) -> dict[str, float]:
    # For some reason, logger.info doesn't work in the WorkloadTask.
    # But it works in the Workload.
    # print(f"_run_workload begins. {self._config=}")
    # The file is at dir /workspace/hello_world_mantaray/test.py.
    # Start training
    import sys
    sys.path.insert(0, "/workspace/mamba")
    import mamba_test
    pretrained_model_name = self._config.workload.model
    d_inner_block_size = self._config.workload.tuning_params['model'][
        "d_inner_block_sizes"
    ]
    max_seq_block_size = self._config.workload.tuning_params['model'][
        "max_seq_block_sizes"
    ]
    batch_size = self._config.workload.tuning_params['model']["batch_sizes"]
    seq_length = self._config.workload.tuning_params['model']["seq_lengths"]
    profile_path = os.path.join(
        self._config.artifact_config.gcs_root,
        self._config.run.experiment_id,
        self._config.run.uid,
    )
    xla_flag = self._config.workload.tuning_params["model"]["libtpu_init_args_space"]
    os.environ["LIBTPU_INIT_ARGS"] = xla_flag

    print(
        f"MantaRay running Mamba task with workload: {pretrained_model_name=} {d_inner_block_size=},"
        f" {max_seq_block_size=}, {batch_size=}, {seq_length=}, {self._config.workload.num_steps=}, {profile_path=}, {os.environ['LIBTPU_INIT_ARGS']=}"
    )
    metrics = mamba_test.run_benchmark(
        pretrained_model_name,
        d_inner_block_size,
        max_seq_block_size,
        batch_size=batch_size,
        seq_length=seq_length,
        repeat=self._config.workload.num_steps,
        profile_path=profile_path,
    )
    print(f"MantaRay finished running Mamba task. {metrics=}")
    return {
        "d_inner_block_size": d_inner_block_size,
        "max_seq_block_size": max_seq_block_size,
        "metrics": metrics,
    }


class MambaWorkload(workload.Workload):

  def __init__(self, config: types.WorkloadConfig) -> None:
    # base_config.workload.tuning_params will return the hyperparameters you set in the job.
    self._config = config

  @property
  def name(self) -> str:
    return "mambaWorkload"

  @property
  def config(self) -> types.WorkloadConfig:
    return self._config

  def get_workload_task(self) -> MambaWorkloadTask:
    return MambaWorkloadTask(self._config)

  def extract_metrics(
      self, results: List[types.WorkloadTaskResult]
  ) -> aotc_metrics.WorkloadMetrics:
    """Extract metrics from Mamba workload."""
    # Sample results=[WorkloadTaskResult(time_taken=134.75280909899993, run_result={'d_inner_block_size': 128, 'max_seq_block_size': 128, 'metrics': {'step_times': [0.8292782306671143, 0.8295426368713379, 0.8295202255249023, 0.8294212818145752]}}), WorkloadTaskResult(time_taken=134.70406099899992, run_result={'d_inner_block_size': 128, 'max_seq_block_size': 128, 'metrics': {'step_times': [0.8291850090026855, 0.8295342922210693, 0.8294618129730225, 0.8294744491577148]}}), WorkloadTaskResult(time_taken=134.71360742399975, run_result={'d_inner_block_size': 128, 'max_seq_block_size': 128, 'metrics': {'step_times': [0.8292276859283447, 0.8295729160308838, 0.8295903205871582, 0.8294034004211426]}}), WorkloadTaskResult(time_taken=134.63977519000036, run_result={'d_inner_block_size': 128, 'max_seq_block_size': 128, 'metrics': {'step_times': [0.8292229175567627, 0.8295581340789795, 0.8295252323150635, 0.8294711112976074]}})]
    metrics = aotc_metrics.get_task_time_metrics(
        [r.time_taken for r in results]
    )

    # only care about rank 0.
    run_result = results[0].run_result
    assert run_result.get('metrics', '') and run_result['metrics'].get('step_times', ''), 'Mamba code failed to return the iteration times.'
    iteration_times = run_result['metrics']['step_times']
    logger.info("Iteration_times : %s", iteration_times)
    metrics.iteration_time = aotc_metrics.MetricStats.from_np(
        np.asarray(iteration_times, dtype=np.float32)
    )
    logger.info(f"Result: {run_result['d_inner_block_size']=}, {run_result['max_seq_block_size']=}, iteration_time={iteration_times}, iteration_time.p50={metrics.iteration_time.p50}")

    return metrics
