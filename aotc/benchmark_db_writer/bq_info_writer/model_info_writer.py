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
TODO: Update model info in the main function & run the script.
"""
import logging
import os

from aotc.benchmark_db_writer import bq_writer_utils
from aotc.benchmark_db_writer.schema.workload_benchmark_v2 import model_info_schema


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def write_model_config(project, dataset, table, dataclass_type,
                       model_id, name, variant, parameter_size_in_billions,
                       update_person_ldap=os.getenv("USER", "mrv2"),
                       description="", details=""):

  writer = bq_writer_utils.create_bq_writer_object(
      project=project,
      dataset=dataset,
      table=table,
      dataclass_type=dataclass_type
  )

  model_info = writer.query(where={"model_id": model_id})
  if model_info:
    raise ValueError("Model id %s is already present in the %s "
                     "table" % (model_id, table))

  # Check if there is already a model info based on name,
  # variant and parameter size
  model_info = writer.query(where={
      "name": name,
      "variant": variant,
      "parameter_size_in_billions": parameter_size_in_billions
  })
  if model_info:
    raise ValueError("Model with name %s, variant %s and "
                     "parameter size %s is already present in the %s "
                     "table" % (name, variant, parameter_size_in_billions,
                                table))

  model_data = model_info_schema.ModelInfo(
      model_id=model_id,
      name=name,
      variant=variant,
      parameter_size_in_billions=parameter_size_in_billions,
      update_person_ldap=update_person_ldap,
      description=description,
      details=details
  )

  logging.info("Writing Data %s to %s table.",
               model_data, table)
  writer.write([model_data])


if __name__ == "__main__":

  project = "ml-workload-benchmarks"
  dataset = "benchmark_dataset_v2"
  table = "model_info"

  # Update it on every run
  model_id = "gemma2-27b"
  name = "Gemma"
  variant = "2"
  parameter_size_in_billions = 27
  description = "https://arxiv.org/pdf/2408.00118"

  write_model_config(
      project=project,
      dataset=dataset,
      table=table,
      model_id=model_id,
      dataclass_type=model_info_schema.ModelInfo,
      name=name,
      variant=variant,
      parameter_size_in_billions=parameter_size_in_billions,
      description=description
  )
