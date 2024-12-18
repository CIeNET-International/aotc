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
"""This script validates the schema of big query table & dataclass present in code.

Modify arguments of check_schema() method in main function
to check prod & test env.
"""

from aotc.benchmark_db_writer.run_summary_writer import sample_run_summary_writer
from aotc.benchmark_db_writer.schema.workload_benchmark_v2 import workload_benchmark_v2_schema


def check_schema(is_test: bool = True) -> None:

  try:
    sample_run_summary_writer.get_db_client(
        "run_summary",
        workload_benchmark_v2_schema.WorkloadBenchmarkV2Schema,
        is_test
    )
    print("No schema mismatch found")
  except Exception as e:
    print("Schema mismatch found.", e)

if __name__ == "__main__":
  # Change is_test flat to True for test env's table
  check_schema(
      is_test=True
  )
