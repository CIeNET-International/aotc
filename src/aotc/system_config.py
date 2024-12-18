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
from typing import Optional
from aotc import types


@dataclasses.dataclass
class TpuConfig:
  tpu_type: str  # e.g. "v5p", "v4"
  # TPUs are allocated via topology not number of nodes/devices
  topology: str  # e.g. 8x8x16
  num_slices: int = 1


@dataclasses.dataclass
class GpuConfig:
  gpu_type: str  # e.g. "H100"
  vm_type: str  # e.g. "A3" or "A3+"
  num_nodes: int
  gpus_per_node: int = 8
  node_type: Optional[str] = None  # This is a key into a3_config
  placement: Optional[str] = None  # e.g. `compact`
  topology_aware: Optional[bool] = True
  max_num_nodes: Optional[int] = None
  # Over-allocate by this many nodes (num_nodes + overallocate) to be able to
  # avoid bad nodes.
  overallocate: Optional[int] = None


def tpu_num_devices_per_slice(tpu_config: TpuConfig) -> int:
  topology = tpu_config.topology
  if isinstance(topology, str):
    topology = [int(d) for d in topology.split("x")]
  return int(np.prod(topology))


def tpu_num_devices(tpu_config: TpuConfig) -> int:
  return tpu_num_devices_per_slice(tpu_config) * tpu_config.num_slices


def tpu_num_nodes(tpu_config: TpuConfig) -> int:
  # Each TPU VM is considered a node
  devices_per_vm = {"v6e": 4, "V6E": 4, "v5p": 4, "V5P": 4, "v4": 4, "V4": 4}
  if tpu_config.tpu_type not in devices_per_vm:
    return NotImplementedError(
        f"TPU type {tpu_config.tpu_type} not yet supported. Please add here."
    )
  num_devices = tpu_num_devices(tpu_config)
  return num_devices // devices_per_vm[tpu_config.tpu_type]


@dataclasses.dataclass
class HardwareConfig:
  accelerator_type: str  # "TPU" or "GPU"
  gpu: Optional[GpuConfig] = None
  tpu: Optional[TpuConfig] = None

  @property
  def num_nodes(self) -> int:
    if self.accelerator_type == "GPU":
      assert self.gpu is not None
      return self.gpu.num_nodes
    elif self.accelerator_type == "TPU":
      return tpu_num_nodes(self.tpu)


def num_devices(hw: HardwareConfig) -> int:
  if hw.accelerator_type == "GPU":
    assert hw.gpu is not None
    if (
        hw.gpu.vm_type == "A3"
        or hw.gpu.vm_type == "A3+"
        or "a3" in hw.gpu.node_type
    ):
      return hw.gpu.num_nodes * 8
    else:
      raise NotImplementedError(
          f"GPU type {hw.gpu.vm_type} not yet supported. Please add here."
      )
  elif hw.accelerator_type == "TPU":
    return tpu_num_devices(hw.tpu)


@dataclasses.dataclass
class SoftwareConfig:
  vm_image: Optional[str] = None  # VM base image
  os_type: Optional[str] = None  # e.g. "debian" or "cos"
  docker_image: Optional[str] = None  # Docker image for the ray worker


@dataclasses.dataclass
class SystemConfig:
  hardware: HardwareConfig
  software: SoftwareConfig

  def system_summary(self, provider: Optional[str] = None) -> types.SystemSummary:
    sw = self.software
    hw = self.hardware
    if hw.tpu:
      acc_type = hw.accelerator_type + ":" + hw.tpu.tpu_type
    else:
      assert hw.gpu is not None
      acc_type = hw.accelerator_type + ":" + hw.gpu.gpu_type
    return types.SystemSummary(
        software=types.SoftwareSystemSummary(
            docker_image=sw.docker_image,
            vm_image=sw.vm_image,
            os_type=sw.os_type,
        ),
        hardware=types.HardwareSystemSummary(
            num_nodes=hw.num_nodes,
            accelerator_type=acc_type,
            num_devices=num_devices(hw),
            node_type=(hw.gpu.node_type if hw.gpu else None),
            topology=(hw.tpu.topology if hw.tpu else None),
            provider=provider,
        ),
    )



