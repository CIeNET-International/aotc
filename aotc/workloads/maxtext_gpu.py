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
from aotc import workload
from aotc import types
import subprocess
import logging
import os
from typing import Any, List, Dict, Optional
from aotc import metric_extraction
from aotc import metrics as aotc_metrics
import traceback


logger = logging.getLogger(__name__)

def get_maxtext_env() -> dict[str, str]:
  return {
      "LD_LIBRARY_PATH": "/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH",
  }

class MaxTextGPUTask(workload.WorkloadTask):

  def __init__(self, config: types.WorkloadConfig):
    self.config = config

  def _prep_node(self,
                 distributed_task_info: types.DistributedTaskInfo) -> None:
    print("Preparing node for MaxText")

  def get_model_config_path(self, model_name):

    # Mapping model ids ( defined in model info table )
    # to gpu model config files in maxtext
    # https://github.com/AI-Hypercomputer/maxtext/tree/main/MaxText/configs/models/gpu
    # the model config file from code instead of reading from repot
    model_path_name_map = {
        "llama2-70b": "llama2_70b.yml",
        "llama2-7b": "llama2_7b.yml",
        "llama3-1-70b": "llama3_70b.yml",
        "llama3-1-8b": "llama3_8b.yml",
        "mixtral-8x1b": "mixtral_8x1b.yml",
    }

    assert (model_name in model_path_name_map.keys()), f"Model {model_name} not yet suppoerted in MaxText"

    file_path = f"/deps/MaxText/configs/models/gpu/{model_path_name_map[model_name]}"
    assert os.path.exists(file_path), f"Model file not found: {file_path}"
    return file_path

  def _run_workload(self,
                    distributed_task_info: types.DistributedTaskInfo) -> dict[str, float]:

    self.config.workload.tuning_params["cli"]["run_name"] = self.config.run.uid
    self.config.workload.tuning_params["cli"]["base_output_directory"] = (
        os.path.join(
          self.config.artifact_config.gcs_root,
          self.config.run.experiment_id)
    )

    if self.config.workload.profiling is not None:
      assert "profiler" in self.config.workload.tuning_params["cli"], (
          "Must specify `profiler` in tuning_params to enable MaxText profiling."
      )
      start_step = self.config.workload.profiling.start_step
      end_step = self.config.workload.profiling.end_step
      self.config.workload.tuning_params["cli"]["skip_first_n_steps_for_profiler"] = start_step
      if end_step is not None:
        self.config.workload.tuning_params["cli"]["profiler_steps"] = end_step - start_step

    model_file_name = self.get_model_config_path(self.config.workload.model)
    maxtext_argv = [f"{k}={v}" for k, v in self.config.workload.tuning_params["cli"].items()]
    print(f"Setting maxtext_argv to {maxtext_argv}")
    xla_flags = [f"{k}={v}" for k, v in self.config.workload.tuning_params["xla"].items()]
    xla_flag_string = " ".join(xla_flags)

    os.environ["XLA_FLAGS"] = xla_flag_string
    print(f"Setting XLA_FLAGS to {xla_flag_string}")

    train_argv = (
        [
            "train.py",
            model_file_name,
            f"steps={self.config.workload.num_steps}",
        ]
        + maxtext_argv
    )


    import sys
    sys.path.insert(0, "/deps/MaxText")

    import train
    # Start training
    print("Starting MaxText training")
    train.main(train_argv)

    # Return config
    import pyconfig
    return pyconfig.config.get_keys()

class MaxTextGPUWorkload(workload.Workload):

  def __init__(self, config: types.WorkloadConfig) -> None:
    self._config = config

  @property
  def name(self) -> str:
    return "MaxText"

  @property
  def config(self) -> types.WorkloadConfig:
    return self._config

  def get_workload_task(self) -> MaxTextGPUTask:
    return MaxTextGPUTask(self._config)

  def _get_tensorboard_metrics(self) -> Optional[aotc_metrics.WorkloadMetrics]:
    tensorboard_dir = self.config.artifact_config.tensorboard_dir + '/' + self.config.run.uid
    if not tensorboard_dir:
      logging.warning(
          "No tensorboard dir specified, skipping metric extraction"
      )
      return None
    extract_config = metric_extraction.tensorboard_metrics.TensorboardMetricExtractionConfig(
        stats_config=self.config.workload.metrics_config,
        iteration_time_metric="perf/step_time_seconds",
        step_name="step",
        throughput_metric="perf/per_device_tflops_per_sec",
    )
    try:
      logging.info('tensorboard_dir %s', tensorboard_dir)
      metrics = metric_extraction.tensorboard_metrics.extract_tensorboard_metrics(
          tensorboard_dir, extract_config=extract_config
      )
      logging.info("Extracted metrics: %s", metrics)
      return metrics
    except Exception as e:
      logging.error(
          "Failed to extract metrics from tensorboard file %s: %s",
          tensorboard_dir,
          repr(e),
      )
      return None

  def extract_metrics(
      self, results: List[types.WorkloadTaskResult]
  ) -> aotc_metrics.WorkloadMetrics:
    metrics = aotc_metrics.get_task_time_metrics([r.time_taken for r in results])

    # only care about rank 0.
    task_result = results[0]
    logger.info("Task Result : %s", task_result)
    model_params = types.ModelParams(
        global_batch_size=task_result.run_result.get("global_batch_size_to_train_on", -1),
        seq_length=task_result.run_result.get("max_target_length", -1),
    )

    # Add some derived metrics
    tokens_per_iter = model_params.global_batch_size * model_params.seq_length
    if metrics.iteration_time and metrics.iteration_time.mean:
      metrics.tokens_per_sec = aotc_metrics.MetricMean(
          mean=tokens_per_iter / metrics.iteration_time.mean
      )
    # Get tensorboard metrics
    try:
      tb_metrics = self._get_tensorboard_metrics()
      if tb_metrics:
        metrics = metrics | tb_metrics
    except Exception as e:  # Corrected exception syntax
      logger.info("Could not extract from tensorboard: %s", e)

     # Add some derived metrics
    tokens_per_iter = model_params.global_batch_size * model_params.seq_length
    if metrics.iteration_time and metrics.iteration_time.mean:
      metrics.tokens_per_sec = aotc_metrics.MetricMean(
          mean=tokens_per_iter / metrics.iteration_time.mean
      )


    metrics.global_batch_size = model_params.global_batch_size
    metrics.seq_length = model_params.seq_length
    # If quantization is not present, return bf16
    if task_result.run_result.get("quantization", ""):
      metrics.precision = task_result.run_result.get("quantization")
    else:
      metrics.precision = "bf16"

    metrics.optimizer = task_result.run_result.get("opt_type", "")

    return metrics
