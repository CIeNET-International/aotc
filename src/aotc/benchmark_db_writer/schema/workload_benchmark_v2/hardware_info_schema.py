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
import datetime
from typing import Optional


@dataclasses.dataclass
class HardwareInfo:
  hardware_id: str
  gcp_accelerator_name: str
  chip_name: str
  bf_16_tflops: int
  memory: float
  hardware_type: str
  provider_name: str
  update_person_ldap: str
  chips_per_node: Optional[int] = None
  description: Optional[str] = ""
  other: Optional[str] = ""
  update_timestamp: datetime.datetime = datetime.datetime.now()
