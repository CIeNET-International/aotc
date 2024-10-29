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
#!/usr/bin/env python
import argparse
import datetime
import logging
import omegaconf
import os
import sys
from aotc.workloads import nemo
from aotc import types


logger = logging.getLogger(__name__)


def run_workload(local_rank: int, num_nodes: int, world_size: int,
                 framework: str, model: str, num_steps: int, timeout: int):
    node_index = int(os.environ["JOB_COMPLETION_INDEX"])
    gpus_per_node = int(os.environ["GPUS_PER_NODE"])
    global_rank = node_index * gpus_per_node + local_rank

    run = types.RunInfo("","",datetime.datetime.now())
    workload = types.WorkloadInfo(framework,model, num_steps,timeout,{})
    artifact_config = types.ArtifactConfig(os.environ["GCS_FUSE_BUCKET"])

    yaml_file = f"/workspace/{framework}.yaml"
    if framework == 'nemo':
        nemo_config = nemo.NemoConfig(omegaconf.OmegaConf.load(yaml_file))
        workload_config = nemo.NemoWorkloadConfig(run, workload, artifact_config, nemo_config)
        task: nemo.NemoWorkloadTask = nemo.NemoWorkloadTask(workload_config)
    else:
        raise NotImplementedError
    distr = types.DistributedTaskInfo(global_rank, num_nodes, world_size, local_rank,int(os.environ["GPUS_PER_NODE"]))
    results = task._run_workload(distr)
    logger.info(f"RESULTS FOR GPU: {global_rank}:\n{results}")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Subprocess worker for running workloads.")
    parser.add_argument("--local_rank", type=int, required=True, help="Local rank of the process.")
    parser.add_argument("--num_nodes", type=int, required=True, help="Total number of nodes.")
    parser.add_argument("--world_size", type=int, required=True, help="Total world size.")
    parser.add_argument("--framework", type=str, required=True, help="Framework to use (e.g., 'nemo').")
    parser.add_argument("--model", type=str, required=True, help="Model name.")
    parser.add_argument("--num_steps", type=int, required=True, help="Number of steps to run.")
    parser.add_argument("--timeout", type=int, required=True, help="Timeout in seconds.")

    args = parser.parse_args()

    run_workload(
        args.local_rank,
        args.num_nodes,
        args.world_size,
        args.framework,
        args.model,
        args.num_steps,
        args.timeout,
    )