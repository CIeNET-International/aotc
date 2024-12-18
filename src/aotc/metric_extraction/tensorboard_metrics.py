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
import logging
from typing import Optional

from aotc import metrics
from aotc import types
from aotc.metric_extraction import tensorboard
import pandas as pd

pd.set_option("display.max_columns", None)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TensorboardMetricExtractionConfig:
  # warmup_iter: int
  iteration_time_metric: str
  step_name: str = "step"
  stats_config: Optional[types.MetricsConfig] = None
  mem_usage_bytes_metric: Optional[str] = None
  throughput_metric: Optional[str] = None
  ignore_columns_with_substr: Optional[str] = None
  loss_metrics: Optional[str] = None
  eval_score_metrics: Optional[list[str]] = None
  convergence_metric: Optional[list[str]] = None


def get_metric_stats(
    df: pd.DataFrame, column_name: str, stats_slice: slice = slice(0, None)
) -> Optional[metrics.MetricStats]:
  logger.info("Column name: %s, Stat slice: %s", column_name, stats_slice)
  values = df[column_name].to_numpy()[stats_slice]
  if values.size == 0:
    return None
  else:
    return metrics.MetricStats.from_np(values)


def get_eval_score_stats(
    df: pd.DataFrame, eval_score_metric: str
) -> Optional[metrics.MetricSimpleStats]:
  if eval_score_metric is None:
    return None
  steps = []
  scores = []
  for index, row in df.iterrows():
    steps.append(int(row["step"]))
    scores.append(row[eval_score_metric])

  eval_score_stats = metrics.MetricIterStats.from_np(steps, scores)
  return eval_score_stats


def extract_tensorboard_metrics(
    tensorboard_dir: str,
    extract_config: TensorboardMetricExtractionConfig,
    df: Optional[pd.DataFrame] = None,
) -> metrics.WorkloadMetrics:
  """Extract metrics from a tensorboard directory.

  Args:
    tensorboard_dir: path to the tensorboard directory
    extract_config: configuration for the extraction
    df: optional dataframe containing the tensorboard data. If not provided, the
      data will be read from the tensorboard_dir.

  Returns:
    WorkloadMetrics object with the extracted metrics
  """
  try:
    reader = tensorboard.TensorboardReader(tensorboard_dir)
    if df is None:
      df = reader.get_scalars_table(
          ignore_columns_with_substr=extract_config.ignore_columns_with_substr
      )
      logger.info("df extracted columns: %s", df.columns)
    # try extracting convergence metrics
    df_convergence = None
    if (
        extract_config.ignore_columns_with_substr
        and extract_config.eval_score_metrics
    ):
      df_convergence = reader.get_iteration_table(
          extract_config.ignore_columns_with_substr
      )
  except Exception as e:
    logger.exception("Failed to extract metrics from tensorboard: %s", repr(e))
    return metrics.WorkloadMetrics()

  logger.info("Metrics DataFrame columns: %s", df.columns)
  logger.info("First 10 rows of metrics:\n%s", df.head(10))

  steps_stats = get_metric_stats(df, extract_config.step_name)
  warmup_iter = None
  if extract_config.stats_config:
    stats_slice = slice(
        extract_config.stats_config.stats_from_step,
        extract_config.stats_config.stats_to_step,
    )
    warmup_iter = extract_config.stats_config.stats_from_step
  else:
    stats_slice = slice(None, None)
  mem_usage_stats = None
  if extract_config.mem_usage_bytes_metric:
    mem_usage_stats = metrics.MetricSimpleStats.from_np(
        df[extract_config.mem_usage_bytes_metric].to_numpy()
    )
  # extract e2e convgence metrics
  eval_score_metric_1, eval_score_metric_2 = None, None
  if (
      extract_config.eval_score_metrics is not None
      and df_convergence is not None
  ):
    out_metrics = []
    if df_convergence is not None and df_convergence.shape[0] > 0:
      for eval_score_metric in extract_config.eval_score_metrics:
        out_metrics.append(
            get_eval_score_stats(df_convergence, eval_score_metric)
        )
      if len(out_metrics) == 1:
        eval_score_metric_1 = out_metrics[0]
      elif len(out_metrics) == 2:
        eval_score_metric_1, eval_score_metric_2 = out_metrics[0], out_metrics[1]

  return metrics.WorkloadMetrics(
      warmup_iter=warmup_iter,
      num_iterations=int(steps_stats.max - steps_stats.min + 1),
      iteration_time=get_metric_stats(
          df,
          extract_config.iteration_time_metric,
          stats_slice=stats_slice,
      ),
      throughput=(
          get_metric_stats(df, extract_config.throughput_metric, stats_slice=stats_slice)
          if extract_config.throughput_metric
          else None
      ),
      mem_usage_bytes=mem_usage_stats,
      loss=(
          get_metric_stats(df, extract_config.loss_metrics)
          if extract_config.loss_metrics
          else None
      ),
      metrics_accuracy_1=eval_score_metric_1,
      metrics_accuracy_2=eval_score_metric_2,
  )
