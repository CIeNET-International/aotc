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
import abc
import logging
import os
import subprocess
import time
from typing import Any, Dict, List

from aotc import metrics
from aotc import types

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# Standalone workload task that is executed on a given node
# Interface for the parts of the workload that run remotely on the worker nodes
class WorkloadTask(abc.ABC):

  # Ray remote functions do not work well with class methods. It is better to
  # use either a class or a function. We use a class in this case.

  @abc.abstractmethod
  def _prep_node(self, task_metadata: types.DistributedTaskInfo) -> None:
    """Runs on every node to prepare the node for running the benchmark"""
    raise NotImplementedError

  @abc.abstractmethod
  def _run_workload(
      self, task_metadata: types.DistributedTaskInfo
  ) -> dict[str, Any]:
    """Runs remotely to execute the workload"""
    raise NotImplementedError

  # Do not override!
  def prep_node_entry(self, task_metadata: types.DistributedTaskInfo) -> None:
    self._prep_node(task_metadata)

  # Do not override!
  def run_workload_entry(
      self,
      artifact_config: types.ArtifactConfig,
      task_metadata: types.DistributedTaskInfo,
  ) -> types.WorkloadTaskResult:

    ex = None
    run_result: Dict[str, Any] = {}
    time_taken: float = 0

    logger.info("Is AOTC code taken from local repo: %s",
            os.environ.get("aotc_from_local"))
    try:
      start = time.perf_counter()
      # run_result = self.run_task(task_metadata=task_metadata)
      run_result = self._run_workload(task_metadata)
      time_taken = time.perf_counter() - start
    except Exception as e:  # pylint: disable=broad-exception-caught
      ex = e
      print(f"Failed to run workload: {e}")

    # Cleanup and artifact collection
    if artifact_config.gcs_enabled:
      self._export_local_artifacts(artifact_config, task_metadata)
    if ex:
      raise ex
    else:
      return types.WorkloadTaskResult(
          time_taken=time_taken, run_result=run_result
      )

  # Do not override!
  def _export_local_artifacts(
      self,
      artifact_config: types.ArtifactConfig,
      task_metadata: types.DistributedTaskInfo,
  ) -> None:
    comm = task_metadata
    # Only the first rank of each node should export artifacts
    if comm.local_rank != 0:
      return
    if os.path.isdir(str(artifact_config.local_dir)) and os.listdir(
        artifact_config.local_dir
    ):
      # Export only if artifact files exist
      files = os.listdir(artifact_config.local_dir)
      print(
          f"Exporting {len(files)} artifacts on rank {comm.rank} from"
          f" {artifact_config.local_dir} to GCS {artifact_config.gcs_dir}, top"
          f" level files are: {str(files)}"
      )

      for f in files:
        local_path = os.path.join(str(artifact_config.local_dir), f)
        if os.path.isdir(local_path) and not os.listdir(local_path):
          print(f"Skipping upload of empty directory local_path {local_path}")
          continue
        print(f"Exporting {f} to GCS {artifact_config.gcs_dir}")
        subprocess.run(
            f'gsutil -m cp -r "{local_path}" "{artifact_config.gcs_dir}/"',
            check=False,
            shell=True,
        )


# Interface for the parts of the workload that execute locally in the controller
class Workload(abc.ABC):

  # NOTE: Order matters! @property must be above @abstractmethod
  @property
  @abc.abstractmethod
  def name(self) -> str:
    raise NotImplementedError

  # NOTE: Order matters! @property must be above @abstractmethod
  @property
  @abc.abstractmethod
  def config(self) -> types.WorkloadConfig:
    raise NotImplementedError

  @abc.abstractmethod
  def get_workload_task(self) -> WorkloadTask:
    raise NotImplementedError

  # TODO - figure out what to about results and its type
  # Right now it contains a list of returns from remotely run ray functions
  @abc.abstractmethod
  def extract_metrics(
      self, results: List[types.WorkloadTaskResult]
  ) -> metrics.WorkloadMetrics:
    """Runs locally"""
    raise NotImplementedError
