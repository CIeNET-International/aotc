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
import copy
import dataclasses
import logging
import omegaconf
import os
import pprint
import traceback
from typing import Any, List, Dict, Optional
import yaml

from aotc import utils
from aotc import gpu_utils
from aotc import workload
from aotc import types
from aotc import metrics as aotc_metrics
from aotc import metric_extraction



logger = logging.getLogger(__name__)


@dataclasses.dataclass
class NemoConfig:
  conf: omegaconf.dictconfig.DictConfig


@dataclasses.dataclass
class NemoWorkloadConfig(types.WorkloadConfig):
  nemo: NemoConfig


def get_nemo_env() -> dict[str, str]:
  return {
      "GLOO_SOCKET_IFNAME": "eth0",
      "NVTE_FWD_LAYERNORM_SM_MARGIN": "8",
      "NVTE_BWD_LAYERNORM_SM_MARGIN": "8",
      "PYTHONFAULTHANDLER": "1",  # print stacktrace on python segfault
  }

# Create WorkloadInfo objects
def create_nemo_workload_info(
    model: str,
    num_steps: int,
    warmup_steps: int,
    profiler_steps: int,
    timeout: int,
    profiler_ranks: list[int] = [0],
):
  return types.WorkloadInfo(
      framework="nemo",
      model=model,
      num_steps=num_steps,
      timeout=timeout,
      tuning_params={},
      runtime_env={},
      profiling=None
      if profiler_steps == 0
      else types.ProfilingConfig(
          start_step=num_steps - profiler_steps - 1,
          end_step=num_steps - 1,
          ranks=profiler_ranks,
      ),
      metrics_config=types.MetricsConfig(
          # Exclude warmup and profiled steps from the summary stats
          stats_from_step=warmup_steps,
          stats_to_step=num_steps - profiler_steps - 1,
      ),
  )


# reads in nemo configuration and overrides values from base_config input argument
def get_nemo_config(base_config: types.WorkloadConfig) -> NemoWorkloadConfig:

  model = base_config.workload.model
  model_yaml = None
  if "NEMO_CONFIG" in os.environ:
    model_yaml = os.environ["NEMO_CONFIG"]
  else:
    model_yaml = os.path.join(
        os.path.dirname(__file__), "nemo_configs", f"{model}.yaml"
    )

  conf = omegaconf.OmegaConf.load(model_yaml)
  conf.run.results_dir = os.path.join(
      base_config.artifact_config.local_dir, "run_results"
  )

  conf.trainer.max_steps = base_config.workload.num_steps

  # Set up profiler
  if base_config.workload.profiling:
    conf.model.nsys_profile.enabled = True
    conf.model.nsys_profile.start_step = (
        base_config.workload.profiling.start_step
    )
    conf.model.nsys_profile.end_step = base_config.workload.profiling.end_step
    conf.model.nsys_profile.ranks = base_config.workload.profiling.ranks
  else:
    conf.model.nsys_profile.enabled = False

  # overwrite model and trainer config based on workload tuning params
  overrides = omegaconf.OmegaConf.create(base_config.workload.tuning_params)
  conf = omegaconf.OmegaConf.merge(conf, overrides)

  if base_config.artifact_config.local_dir:
    conf.exp_manager.explicit_log_dir = os.path.join(
        base_config.artifact_config.local_dir, "log"
    )
    conf.exp_manager.exp_dir = base_config.artifact_config.local_dir
  base_config.artifact_config.tensorboard_dir = (
      base_config.artifact_config.gcs_dir + "/log"
  )

  return NemoWorkloadConfig(
      **base_config.shallow_dict(),
      nemo=NemoConfig(conf=conf),
  )


# adapted from Megatron-LM/megatron/training/training.py
def num_floating_point_operations(
    transformer, batch_size, seq_length, padded_vocab_size
):
  """Args:

    transformer: megatron.core.transformer.transformer_config.TransformerConfig
    batch_size: global batch size
    seq_length: sequence length
    padded_vocab_size: padded vocab size

  Returns:
    number of floating point operations of the model, does NOT count remat.
  """
  import torch.nn.functional as F
  # Attention projection size.
  query_projection_size = (
      transformer.kv_channels * transformer.num_attention_heads
  )
  query_projection_to_hidden_size_ratio = (
      query_projection_size / transformer.hidden_size
  )
  # Group Query Attention.
  # MoE.
  num_experts_routed_to = (
      1 if transformer.num_moe_experts is None else transformer.moe_router_topk
  )
  gated_linear_multiplier = (
      3 / 2 if transformer.activation_func == F.silu else 1
  )
  return (
      12
      * batch_size
      * seq_length
      * transformer.num_layers
      * transformer.hidden_size
      * transformer.hidden_size
      * (
          # Attention.
          (
              (
                  1
                  + (
                      transformer.num_query_groups
                      / transformer.num_attention_heads
                  )
                  + (seq_length / transformer.hidden_size)
              )
              * query_projection_to_hidden_size_ratio
          )
          # MLP.
          + (
              (transformer.ffn_hidden_size / transformer.hidden_size)
              * num_experts_routed_to
              * gated_linear_multiplier
          )
          # Logit.
          + (
              padded_vocab_size
              / (2 * transformer.num_layers * transformer.hidden_size)
          )
      )
  )


class NemoWorkloadTask(workload.WorkloadTask):

  def __init__(self, config: NemoWorkloadConfig):
    self._config: NemoWorkloadConfig = config

  def _prep_node(self, distributed_task_info: types.DistributedTaskInfo) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("Preparing node for NeMo")
    self._predownload_vocab()
    logging.info("Vocab downloaded")

    gpu_utils.enable_nvidia_persistence()

    conf = self._config.nemo.conf
    if self._config.artifact_config.local_dir:
      assert conf.exp_manager.explicit_log_dir
      os.makedirs(conf.exp_manager.explicit_log_dir, exist_ok=True)
      os.makedirs(conf.run.results_dir, exist_ok=True)

  def _predownload_vocab(self):
    conf = self._config.nemo.conf
    if conf.model.tokenizer.vocab_file == "gpt2-vocab.json":
      assert conf.model.tokenizer.merge_file == "gpt2-merges.txt"
      utils.retry_download(
          "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
          "gpt2-vocab.json",
      )
      utils.retry_download(
          "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
          "gpt2-merges.txt",
      )
    else:
      logging.warning(
          "Not downloading because vocab file is not gpt2: %s",
          conf.model.tokenizer.vocab_file,
      )

  def _dump_configs(self, local_dir: str) -> None:

    with open(os.path.join(local_dir, "workload_config.yaml"), "w") as f:
      config_wo_nemo = copy.deepcopy(self._config)
      config_wo_nemo.nemo = None
      yaml.dump(config_wo_nemo, f)
    with open(os.path.join(local_dir, "environ.txt"), "w") as f:
      f.write(pprint.pformat(os.environ))
    omegaconf.OmegaConf.save(
        config=self._config.nemo.conf,
        f=os.path.join(local_dir, "nemo_config.yaml"),
    )

  def _run_workload(self, distributed_task_info: types.DistributedTaskInfo) -> dict[str, float]:

    cfg = self._config.nemo.conf
    cfg.trainer.devices = distributed_task_info.num_tasks_per_node
    cfg.trainer.num_nodes = distributed_task_info.num_nodes

    local_dir = self._config.artifact_config.local_dir
    if local_dir and not os.path.exists(local_dir):
      os.makedirs(local_dir, exist_ok=True)

    subdirs = ["tensorboard", "profile"]
    for d in subdirs:
      os.makedirs(os.path.join(str(local_dir), d), exist_ok=True)

    if local_dir and distributed_task_info.rank == 0:
      self._dump_configs(local_dir)

    gpus_per_nodes = os.environ.get("GPUS_PER_NODE")
    if not gpus_per_nodes:
      raise ValueError("GPUS_PER_NODE not set")

    # By default, Ray will set CUDA_VISIBLE_DEVICES to 1 GPU per task, but NeMo
    # and Lightning require all GPUs to be visible for successful initialization.
    # NOTE: it is important to set this before importing torch (either directly
    # or indirectly from the nemo imports). This is because the CUDA_VISIBLE_DEVICES
    # is read during torch import and modifying it afterwards has no effect.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(i) for i in range(int(gpus_per_nodes))
    )

    import time
    import random
    # Workaround for crashes in `import torch` that flakily crashes with segfaults,
    # which I suspect are caused by a race-condition while loading torch shared
    # libraries
    # Note quite sure what is happening, but the random seems to help
    # time.sleep(random.uniform(2,10))
    # time.sleep(4 * (rank % tasks_per_node))
    time.sleep(4*distributed_task_info.local_rank)

    os.environ["LOCAL_RANK"] = str(distributed_task_info.local_rank)
    os.environ["OMP_NUM_THREADS"]=str(12)


    import torch
    import torch.multiprocessing as mp
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
    from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
    from nemo.utils.exp_manager import exp_manager

    torch._dynamo.config.suppress_errors = True
    mp.set_start_method("spawn", force=True)

    # NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py
    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)
    model = MegatronGPTModel(cfg.model, trainer)

    # compute flops
    seq_length = cfg.model.encoder_seq_length
    batch_size = cfg.model.global_batch_size
    padded_vocab_size = model.padded_vocab_size
    flops = num_floating_point_operations(
        model.transformer_config,
        batch_size=batch_size,
        seq_length=seq_length,
        padded_vocab_size=padded_vocab_size,
    )

    trainer.fit(model)
    return {
        "flops": flops,
        "global_batch_size": batch_size,
        "seq_length": seq_length,
    }


class NemoWorkload(workload.Workload):

  def __init__(self, config: types.WorkloadConfig) -> None:
    self._config: NemoWorkloadConfig = get_nemo_config(config)

  @property
  def name(self) -> str:
    return "nemo"

  @property
  def config(self) -> NemoWorkloadConfig:
    return self._config

  def get_workload_task(self) -> NemoWorkloadTask:
    return NemoWorkloadTask(self._config)

  def _get_tensorboard_metrics(
      self, flops: Optional[float] = None, world_size: Optional[int] = None
  ) -> Optional[aotc_metrics.WorkloadMetrics]:
    # tensorboard_dir = self.config.artifact_config.tensorboard_dir
    tensorboard_dir = str(self.config.artifact_config.gcs_dir) + "/log"
    if not tensorboard_dir:
      logging.warning(
          "No tensorboard dir specified, skipping metric extraction"
      )
      return None
    extract_config = metric_extraction.tensorboard_metrics.TensorboardMetricExtractionConfig(
        stats_config=self.config.workload.metrics_config,
        iteration_time_metric="train_step_timing in s",
        step_name="step",
    )
    try:
      reader = metric_extraction.tensorboard.TensorboardReader(tensorboard_dir)
      df = reader.get_scalars_table()
      if flops:
        # augment with "throughput"
        df["throughput"] = flops / (
            df["train_step_timing in s"] * 10**12 * world_size
        )
        extract_config.throughput_metric = "throughput"

      metrics = metric_extraction.tensorboard_metrics.extract_tensorboard_metrics(
          tensorboard_dir, extract_config=extract_config, df=df
      )
      logging.info("Extracted metrics: %s", metrics)
      return metrics
    except Exception as e:
      logging.error(
          "Failed to extract metrics from tensorboard file %s: %s",
          tensorboard_dir,
          repr(e),
      )
      logging.error(f"Traceback:\n{traceback.format_exc()}")
      return None


  def extract_metrics(
      self, results: List[types.WorkloadTaskResult]
  ) -> aotc_metrics.WorkloadMetrics:
    """From nemo.NemoPretrainingWorkload.get_metrics()"""
    metrics = aotc_metrics.get_task_time_metrics([r.time_taken for r in results])

    # only care about rank 0.
    task_result = results[0]
    model_params = types.ModelParams(
        global_batch_size=task_result.run_result.get("global_batch_size",1),
        seq_length=task_result.run_result.get("seq_length",1),
    )
    world_size = task_result.run_result.get("world_size",1)

    # Get tensorboard metrics
    tb_metrics = self._get_tensorboard_metrics(
        flops=task_result.run_result.get("flops",0), world_size=world_size
    )
    if tb_metrics:
      metrics = metrics | tb_metrics

    # Add some derived metrics
    tokens_per_iter = model_params.global_batch_size * model_params.seq_length
    if metrics.iteration_time and metrics.iteration_time.mean:
      metrics.tokens_per_sec = aotc_metrics.MetricMean(
          mean=tokens_per_iter / metrics.iteration_time.mean
      )
    return metrics
