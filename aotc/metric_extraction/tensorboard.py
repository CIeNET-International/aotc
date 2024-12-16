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

"""Simple class to aid reading metrics from tensorboard."""
import logging
from typing import Optional

import pandas as pd
# from tensorflow.core.util import event_pb2
from tensorboard.backend.event_processing import event_accumulator

logger = logging.getLogger(__name__)

# requires either:
# 1) tensorflow or (fsspec and gcsfs)


class TensorboardReader:
  """Simple class that loads the results from a tensorboard log directory and returns the per-step metrics as a pandas dataframe."""

  def __init__(self, path):
    """Loads the results at the given path or returns an exception."""
    self.event_acc = event_accumulator.EventAccumulator(path)
    self.event_acc.Reload()
    logger.info("Path given to TensorboardReader: %s", path)

  # per step metrics
  def get_scalars_table(self, ignore_columns_with_substr: Optional[str] = None):
    """Returns the scalar metrics of the tensorboard as a pandas dataframe."""
    row_by_step = {}
    self.event_acc.Reload()
    logger.info("Scalers: %s", self.event_acc.Tags())
    for metric in self.event_acc.Tags()["scalars"]:
      logger.info("metric: %s", metric)
      if ignore_columns_with_substr and ignore_columns_with_substr in metric:
        continue
      event_list = self.event_acc.Scalars(metric)
      for event in event_list:
        row_by_step.setdefault(event.step, {"step": event.step})[
            metric
        ] = event.value
    list_of_rows = list(row_by_step.values())
    list_of_rows.sort(key=lambda row: row["step"])
    return pd.DataFrame(data=list_of_rows)

  # per iteration metrics
  def get_iteration_table(self, metrics_to_extract: Optional[str] = None):
    """Returns the scalar metrics of the tensorboard as a pandas dataframe."""
    row_by_step = {}
    for metric in self.event_acc.Tags()["scalars"]:
      if metrics_to_extract and metrics_to_extract in metric:
        event_list = self.event_acc.Scalars(metric)
        for event in event_list:
          print(event.step)
          row_by_step.setdefault(event.step, {"step": event.step})[
              metric
          ] = event.value
    list_of_rows = list(row_by_step.values())
    list_of_rows.sort(key=lambda row: row["step"])
    return pd.DataFrame(data=list_of_rows)
