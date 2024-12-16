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
"""
TODO: Update hardware info in the main function & run the script.
"""
import logging
import os

from aotc.benchmark_db_writer import bq_writer_utils
from aotc.benchmark_db_writer.schema.workload_benchmark_v2 import hardware_info_schema


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_hardware_config(project, dataset, table, dataclass_type,
                          hardware_id, gcp_accelerator_name, chip_name,
                          bf_16_tflops, memory, hardware_type, provider_name,
                          chips_per_node=None,
                          update_person_ldap=os.getenv("USER", "mrv2"),
                          description="", other=""):

  writer = bq_writer_utils.create_bq_writer_object(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=dataclass_type
  )

  hardware_info = writer.query(where={"hardware_id": hardware_id})
  if hardware_info:
    raise ValueError("Hardware id %s is already present in the %s "
                     "table" % (hardware_id, table))

  hardware_data = hardware_info_schema.HardwareInfo(
      hardware_id=hardware_id,
      gcp_accelerator_name=gcp_accelerator_name,
      chip_name=chip_name,
      bf_16_tflops=bf_16_tflops,
      memory=memory,
      chips_per_node=chips_per_node,
      hardware_type=hardware_type,
      provider_name=provider_name,
      update_person_ldap=update_person_ldap,
      description=description,
      other=other
  )

  logging.info("Writing Data %s to %s table.",
               hardware_data, table)
  writer.write([hardware_data])


if __name__ == "__main__":

  project = "ml-workload-benchmarks"
  dataset = "benchmark_dataset_v2"
  table = "hardware_info"

  # Update it on every run
  hardware_id = "a3ultra"
  gcp_accelerator_name = "A3Ultra"
  chip_name = "H200"
  bf_16_tflops = 989
  memory = 141
  chips_per_node = None
  hardware_type = "GPU"
  provider_name = "Nvidia"
  description = ""

  write_hardware_config(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=hardware_info_schema.HardwareInfo,
      hardware_id=hardware_id,
      gcp_accelerator_name=gcp_accelerator_name,
      chip_name=chip_name,
      bf_16_tflops=bf_16_tflops,
      memory=memory,
      chips_per_node=chips_per_node,
      description=description,
      hardware_type=hardware_type,
      provider_name=provider_name
  )
