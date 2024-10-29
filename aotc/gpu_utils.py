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
import logging
import os
import subprocess
from typing import Any, Dict


def enable_nvidia_persistence():
  result = subprocess.run(["nvidia-smi", "-pm", "1"], capture_output=True)
  if result.returncode != 0:
    raise RuntimeError(result.stderr)
  else:
    logging.info("Nvidia persistence enabled, output: %s", result.stdout)
  return result.stdout


def get_interface_mapping(os_type: str, vm_type: str):
  assert os_type
  if os_type != "debian":
    return {}
  if vm_type == "a3plus" or vm_type == "A3+":
    return {
        "eth0": "enp0s12",
        "eth1": "enp6s0f0",
        "eth2": "enp7s0f0",
        "eth3": "enp13s0f0",
        "eth4": "enp14s0f0",
        "eth5": "enp134s0f0",
        "eth6": "enp135s0f0",
        "eth7": "enp141s0f0",
        "eth8": "enp142s0f0",
    }
  elif vm_type == "a3" or vm_type == "A3":
    return {
        "eth0": "enp0s12",
        "eth1": "enp6s0",
        "eth2": "enp12s0",
        "eth3": "enp134s0",
        "eth4": "enp140s0",
    }
  else:
    raise ValueError(f"Unknown VM type: {vm_type}")

def load_nccl_env(nccl_env_vars: Dict[str,Any], os_type: str, vm_type: str):
  if_mapping = get_interface_mapping(
      os_type=os_type, vm_type=vm_type
  )

  if if_mapping:
    for var, value in nccl_env_vars.items():
      if "eth" in value:
        for eth, intf in if_mapping.items():
          value = value.replace(eth, intf)
        nccl_env_vars[var] = value

  return nccl_env_vars

def get_nccl_env(nccl_env_vars: Dict[str,Any], os_type: str, vm_type: str, nccl_debug_log=True):
  nccl_env = load_nccl_env(nccl_env_vars, os_type, vm_type)
  nccl_logging = {}
  if nccl_debug_log:
    nccl_logging = {
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,ENV,TUNING,NET,VERSION",
    }
  return {**nccl_env, **nccl_logging}
