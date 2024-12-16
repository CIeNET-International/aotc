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
class ModelInfo:
  model_id: str
  name: str
  variant: str
  parameter_size_in_billions: int
  update_person_ldap: str
  description: Optional[str] = ""
  details: Optional[str] = ""
  update_timestamp: datetime.datetime = datetime.datetime.now()
