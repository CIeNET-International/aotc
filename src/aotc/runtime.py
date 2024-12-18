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
from typing import Any, Callable, Optional, Union

from aotc import types

@dataclasses.dataclass
class ProfilerOptions:
  profile_ranks: list[int] = dataclasses.field(default_factory=list)
  output_dir: str = ""


RuntimeEnvFactory = Callable[[types.DistributedTaskInfo], dict[str, Any]]
""" Factory function for runtime_env."""

RuntimeEnvType = Union[RuntimeEnvFactory, dict[str, Any]]
"""Type definition of a runtime_env parameter.

This can be either directly the runtime_env dictionary or a function that
given a task's DistributedRuntimeInfo, returns the runtime_env dictionary.
"""
