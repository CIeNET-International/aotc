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
import argparse
import datetime
import json
import logging
import multiprocessing
import os
import subprocess
import time

from typing import Dict, Optional
from aotc import types
from aotc.workloads import nemo
from aotc import workload
from aotc import workload_utils
from aotc import system_config
from aotc import gpu_utils

logger = logging.getLogger(__name__)


class StandaloneExecutor:

  def __init__(self):
    self.standalone_task: workload.WorkloadTask

    # TODO: read in values from somewhere, environment or args
    # Maybe once a Mantaray run starts, it can generate a yaml with all these
    # values that can be added to helm template and read in

    ##################### Workload Info #####################
    framework = os.environ["FRAMEWORK"]
    model = os.environ["MODEL"]
    num_steps = int(os.environ["NUM_STEPS"])
    warmup_steps = int(os.environ["WARMUP_STEPS"])
    profiler_steps = int(os.environ["PROFILER_STEPS"])
    workload_arguments = os.environ.get("WORKLOAD_ARGUMENTS", "").split()

    ###################### Job Config #######################
    config_dir = os.environ["JOB_CONFIG_DIR"]
    config_name = os.environ["JOB_CONFIG_NAME"]
    name_prefix = os.environ["JOB_NAME_PREFIX"]

    ##################### System Config #####################
    num_nodes = int(os.environ["NUM_NODES"])
    accelerator_type = os.environ["ACCELERATOR_TYPE"]
    gpu_type = os.environ["GPU_TYPE"]
    vm_type = os.environ["VM_TYPE"]
    node_type = os.environ["NODE_TYPE"]
    docker_image = os.environ["DOCKER_IMAGE"]
    os_type = os.environ["OS_TYPE"]

    ################### Experiment Config ###################
    exp_name = os.environ["EXP_NAME"]
    output_config_path = os.getenv("OUTPUT_CONFIG_PATH",os.path.abspath(
      "/workspace/output.yaml"
      )
    )
    self.now = datetime.datetime.fromtimestamp(int(os.environ["JOB_TIMESTAMP"]))
    # create experiment config
    self.experiment_config = workload_utils.experiment_config(
      name=exp_name,
      output_config_path=output_config_path,
      skip_existing=False,
    )

    # Create System Config Object
    self.system_config = system_config.SystemConfig(
      hardware=system_config.HardwareConfig(
        accelerator_type=accelerator_type,
        gpu=system_config.GpuConfig(
          gpu_type=gpu_type,
          vm_type=vm_type,
          node_type=node_type,
          num_nodes=num_nodes,
        ),
      ),
      software=system_config.SoftwareConfig(
        docker_image=docker_image,
        os_type=os_type,
      ),
    )
    # Create Artifact Config Object
    gcs_root = "gs://" + os.getenv("GCS_BUCKET","")
    gcs_enabled = bool(os.getenv("GCS_ENABLED", True))
    gcs_dir = os.path.join(gcs_root,os.getenv("GCS_DIR",f"{exp_name}/{self.now.strftime('%Y-%m-%d_%H:%M:%S')}"))
    tensorboard_dir = os.getenv("TENSORBOARD_DIR")
    local_root = os.getenv("LOCAL_ROOT","/tmp/workload-artifacts")
    local_dir = os.getenv("LOCAL_DIR",f"{local_root}/results")
    tensorboard_tmp = bool(os.getenv("TENSORBOARD_TEMP", True))
    tensorboard_rank = int(os.getenv("TENSORBOARD_RANK",0))
    json_str = os.environ.get("COLLECT_DIRS","{}")
    collect_dirs = json.loads(json_str)

    self.artifact_config = types.ArtifactConfig(
      gcs_root=gcs_root,
      gcs_enabled=gcs_enabled,
      gcs_dir=gcs_dir,
      tensorboard_dir=tensorboard_dir,
      local_root=local_root,
      local_dir=local_dir,
      tensorboard_tmp=tensorboard_tmp,
      tensorboard_rank=tensorboard_rank,
      collect_dirs=collect_dirs
    )

    if framework == "nemo":
      # Create workload_info object
      self.workload_info: types.WorkloadInfo = nemo.create_nemo_workload_info(
        model=model,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        profiler_steps=profiler_steps,
        timeout=num_steps * 240,
        profiler_ranks=workload_utils.profile_ranks_staggered(
          num_nodes=num_nodes, num_all=2
        ),
      )
      # Update workload_info with additional settings
      env_vars = self._create_env_for_subprocess()

      self.workload_info.runtime_env = {
          "env_vars": env_vars,
      }

      # Base workload config
      self.base_workload_config: types.WorkloadConfig = workload_utils.create_workload_config(self.workload_info,
                                                                          self.experiment_config.output.artifact_config,
                                                                          exp_name,
                                                                          self.now)

      # create standlone task
      self.workload = nemo.NemoWorkload(self.base_workload_config)
      self.standalone_task = self.workload.get_workload_task()

    else:
      raise NotImplementedError

  def _update_env_with_artifact_config_settings(self,env) -> Dict[str, str]:
    env["GCS_ROOT"]=self.artifact_config.gcs_root
    env["GCS_ENABLED"]=str(self.artifact_config.gcs_enabled)
    env["GCS_DIR"]=self.artifact_config.gcs_dir
    if self.artifact_config.tensorboard_dir:
      env["TENSORBOARD_DIR"]=self.artifact_config.tensorboard_dir
    env["LOCAL_ROOT"]=self.artifact_config.local_root
    env["LOCAL_DIR"]=self.artifact_config.local_dir
    env["TENSORBOARD_TMP"]=str(self.artifact_config.tensorboard_tmp)
    env["TENSORBOARD_RANK"]=str(self.artifact_config.tensorboard_rank)
    env["COLLECT_DIRS"]=json.dumps(self.artifact_config.collect_dirs)
    return env


  # TODO: Push results to bucket
  def _log_result(self, result: Dict[str,float]):
    logger.info("_log_result not Implemented")

  def _prep_node(self, task_metadata: types.DistributedTaskInfo):
    if self.standalone_task:
      self.standalone_task.prep_node_entry(task_metadata)
    else:
      raise NotImplementedError

  def _create_env_for_subprocess(self) -> Dict[str, str]:
    # Read environment variables from the file
    env = {}
    with open(os.environ["ENV_FILE"], "r") as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            env[key] = value
    env = self._update_env_with_artifact_config_settings(env)
    return env

  def run(self):
    # read environment variables
    # TODO: We may need to do something different for JAX on GPU.
    # TODO: Will need TPU standalone executor as well probably.
    local_rank = int(os.getenv("LOCAL_RANK",0))
    gpus_per_node = int(os.environ["GPUS_PER_NODE"])
    node_index = int(os.environ["JOB_COMPLETION_INDEX"])
    rank = (node_index * gpus_per_node) + local_rank

    world_size = int(os.getenv("WORLD_SIZE",1))
    num_nodes = int(os.getenv("NNODES",1))

    # Create Prep Node Metadata
    num_tasks_per_node = int(os.getenv("NUM_PREP_TASKS_PER_NODE",1))
    prep_node_task_metadata = types.DistributedTaskInfo(rank,num_nodes,world_size,local_rank)
    self._prep_node(prep_node_task_metadata)

    # Create Run Workload Metadate
    num_tasks_per_node = int(os.getenv("NUM_WORKLOAD_TASKS_PER_NODE",8))
    run_workload_task_metadata = types.DistributedTaskInfo(rank,num_nodes,world_size,local_rank, num_tasks_per_node)

    # Create environment for all subprocesses
    env = self._create_env_for_subprocess()


    processes = []
    for local_rank in range(gpus_per_node):
      command = [
          "python",
          "-m",
          "aotc.subprocess_worker",
          "--local_rank", str(local_rank),
          "--num_nodes", str(num_nodes),
          "--world_size", str(world_size),
          "--framework", self.workload_info.framework,
          "--model", self.workload_info.model,
          "--num_steps", str(self.workload_info.num_steps),
          "--timeout", str(self.workload_info.timeout)
      ]
      env["RANK"] = str((node_index * gpus_per_node) + local_rank)
      env["NODE_RANK"] = str(node_index)
      logger.info(f"Launching subprocess for GPU: {local_rank}")

      process = subprocess.Popen(command, env=env)
      processes.append(process)
      logger.info(f"Launched {process.pid} for GPU: {local_rank}")

    # Wait for processes to finish
    start = time.perf_counter()
    for process in processes:
      process.wait()
    time_taken = time.perf_counter() - start
    logger.info(f"Workload complete. Time taken: {time_taken}")
    return


def main():
  executor = StandaloneExecutor()
  executor.run()

if __name__ == "__main__":
    main()