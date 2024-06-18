# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Pre-training a Llama2-style LLM with Mosaic MCT
# MAGIC
# MAGIC In this demo we will cover how to create a pre-training run with Databricks Mosaic MCT

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding the Run Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run Configuration with Mosaic CLI for MCT Pre-Training
# MAGIC
# MAGIC Run submissions to the MCT platform can be configured through a YAML file or using our Python API.
# MAGIC
# MAGIC The fields are identical across both methods:
# MAGIC | Field           |          | Type                                              |
# MAGIC | --------------- | -------- | ------------------------------------------------- |
# MAGIC | `name`          | required | `str`                                             |
# MAGIC | `image`         | required | `str`                                             |
# MAGIC | `command`       | required | `str`                                             |
# MAGIC | `compute`       | required | {class}`~mcli.models.run_config.ComputeConfig`    |
# MAGIC | `scheduling`    | optional | {class}`~mcli.models.run_config.SchedulingConfig` |
# MAGIC | `integrations`  | optional | `List[Dict]`                                      |
# MAGIC | `env_variables` | optional | `Dict[str, str]`                                  |
# MAGIC | `parameters`    | optional | `Dict[str, Any]`                                  |
# MAGIC | `metadata`      | optional | `Dict[str, Any]`                                  |
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-Training Configuration Field Types
# MAGIC
# MAGIC #### `name`
# MAGIC
# MAGIC A run name is the primary identifier for working with runs.
# MAGIC For each run, a unique identifier is automatically appended to the provided run name.
# MAGIC After submitting a run, the finalized unique name is displayed in the terminal, and can also be viewed with `mcli get runs` or {class}`~mcli.Run` object.
# MAGIC
# MAGIC #### `image`
# MAGIC
# MAGIC Runs are executed within [Docker containers](https://docs.docker.com/get-started/overview/#containers) defined by a
# MAGIC [Docker image](https://docs.docker.com/get-started/overview/#images).
# MAGIC
# MAGIC #### `command`
# MAGIC
# MAGIC The command is what's executed when the run starts, typically to launch your training jobs and scripts. For example, the following command:
# MAGIC
# MAGIC ```
# MAGIC command: |
# MAGIC   echo Hello World!
# MAGIC ```
# MAGIC will result in a run that prints "Hello World" to the console.
# MAGIC
# MAGIC In many cases, such as the example we will show, this is also where the data preparation can take place if the data is not yet in a trainable state. 
# MAGIC
# MAGIC If you are training models with [Composer](https://docs.MCT.com/projects/composer), then the `command` field is where you will write
# MAGIC your [Composer launch command](https://docs.MCT.com/projects/composer/en/latest/trainer/using_the_trainer.html#distributed-training).
# MAGIC
# MAGIC #### `compute` 
# MAGIC
# MAGIC The compute field specifies which compute resources to request for your run.
# MAGIC The MCT platform will try and infer which compute resources to use automatically.
# MAGIC Which fields are required depend on which and what type of clusters are available to your organization.
# MAGIC If those resources are not valid or if there are multiple options still available, an error will be raised on run submissions, and the run will not be created.
# MAGIC
# MAGIC | Field        | Type        | Details                                                                                               |
# MAGIC | ------------ | ----------- | ----------------------------------------------------------------------------------------------------- |
# MAGIC | `gpus`       | `int`       | Typically required, unless you specify `nodes` or a cpu-only run                                      |
# MAGIC | `cluster`    | `str`       | Required if you have multiple clusters                                                                |
# MAGIC | `gpu_type`   | `str`       | Optional                                                                                              |
# MAGIC | `instance`   | `str`       | Optional. Only needed if the cluster has multiple GPU instances                                       |
# MAGIC | `cpus`       | `int`       | Optional. Typically not used other than for debugging small deployments.                              |
# MAGIC | `nodes`      | `int`       | Optional. Alternative to `gpus` - typically there are 8 GPUs per node                                 |
# MAGIC | `node_names` | 'List[str]` | Optional. Names of the nodes to use in the run. You can find these via `mcli describe cluster <name>` |
# MAGIC
# MAGIC You can see clusters, instances, and compute resources available to you using:
# MAGIC
# MAGIC ```
# MAGIC mcli get clusters
# MAGIC ```
# MAGIC
# MAGIC For example, you can launch a multi-node cluster `my-cluster` with 16 A100 GPUs:
# MAGIC
# MAGIC ```yaml
# MAGIC compute:
# MAGIC   cluster: my-cluster
# MAGIC   gpus: 16
# MAGIC   gpu_type: a100_80gb
# MAGIC ```
# MAGIC
# MAGIC #### 'scheduling`
# MAGIC
# MAGIC The `scheduling` field governs how the MCT platform's scheduler will manage your run.
# MAGIC It is a simple dictionary, currently containing one key: `priority`.
# MAGIC
# MAGIC | Field                     |          | Type    |
# MAGIC | ------------------------- | -------- | ------- |
# MAGIC | `priority`                | optional | `str`   |
# MAGIC | `preemptible`             | optional | `bool`  |
# MAGIC | `max_retries`             | optional | `int`   |
# MAGIC | `retry_on_system_failure` | optional | `bool`  |
# MAGIC | `max_duration`            | optional | `float` |
# MAGIC
# MAGIC **priority**: Runs in the platform's scheduling queue are first sorted by their priority, then by their creation time.
# MAGIC The `priority` field defaults to `auto` when omitted or used with unavailable options.
# MAGIC It can be overridden to `low` or `lowest` to allow other runs higher priority.
# MAGIC When using a reserved cluster, best practices usually dictate that large numbers of more experimental runs (think
# MAGIC exploratory hyperparameter sweeps) should be run at `low` priority to allow your teammates' runs higher priority,
# MAGIC whereas important "hero" runs should be run at the default `auto` priority. When in a shared cluster, we recommend leaving the
# MAGIC default set to `auto` to schedule as fast as possible on the shared cluster
# MAGIC will queue above your own.
# MAGIC
# MAGIC **preemptible**: If your run can be retried, you can set `preemptible` to `True`.
# MAGIC
# MAGIC **max_retries**: This is the maximum number of times our system will attempt to retry your run.
# MAGIC
# MAGIC **retry_on_system_failure**: If you want your run to be retried if it encounters a system failure, you can set
# MAGIC retry_on_system_failure to `True`
# MAGIC
# MAGIC **max_duration**: This is the time duration (in hours) that your run can run for before it is stopped.
# MAGIC
# MAGIC #### `integrations`
# MAGIC
# MAGIC We support many [Integrations](page:integrations/integration_basics) to customize aspects of both the run setup and environment.
# MAGIC
# MAGIC Integrations are specified as a list in the YAML. Each item in the list must specify a valid `integration_type`
# MAGIC along with the relevant fields for the requested integration.
# MAGIC
# MAGIC Some examples of integrations include automatically cloning a [Github](https://github.com) repository,
# MAGIC installing python packages, and setting up logging to an [MLflow](https://www.databricks.com/product/managed-mlflow) experiment are shown below:
# MAGIC
# MAGIC ```yaml
# MAGIC integrations:
# MAGIC   - integration_type: git_repo
# MAGIC     git_repo: org/my_repo
# MAGIC     git_branch: my-work-branch
# MAGIC   - integration_type: pip_packages
# MAGIC     packages:
# MAGIC       - numpy>=1.22.1
# MAGIC       - requests
# MAGIC   - integration_type: mlflow
# MAGIC     project: my_weight_and_biases_project
# MAGIC     entity: mosaicml
# MAGIC ```
# MAGIC
# MAGIC You can **read more about integrations on the [Integrations Page](page:integrations/integration_basics).**
# MAGIC
# MAGIC Some integrations may require adding secrets. For example, pulling from a private github repository would require the `git-ssh` secret to be configured.
# MAGIC See the [Secrets Page](page:secrets/secret_basics).
# MAGIC
# MAGIC #### `env_variables`
# MAGIC
# MAGIC Environment variables can also be injected into each run at runtime through the `env_variables` field.
# MAGIC
# MAGIC For example, the below YAML will print "Hello MOSAICML my name is MOSAICML_TWO!":
# MAGIC
# MAGIC ```yaml
# MAGIC name: hello-world
# MAGIC image: python
# MAGIC command: |
# MAGIC   sleep 2
# MAGIC   echo Hello $NAME my name is $SECOND_NAME!
# MAGIC env_variables:
# MAGIC   NAME: MOSAICML
# MAGIC   SECOND_NAME: MOSAICML_TWO
# MAGIC ```
# MAGIC
# MAGIC The `command` accesses the value of the environment variable by the key (in this case `$NAME` and `$SECOND_NAME`).
# MAGIC
# MAGIC ```
# MAGIC This configuration is not intended to be used for sensitive environment variables such as api keys, tokens, or passwords. Please configure these environment variables via [environment variable secrets](../resources/secrets/env.md)
# MAGIC ```
# MAGIC
# MAGIC #### `parameters` 
# MAGIC
# MAGIC The provided parameters are mounted as a YAML file of your run at `/mnt/config/parameters.yaml` for your code to access.
# MAGIC Parameters are a popular way to easily configure your training run. see **Model Parameters** (below).
# MAGIC
# MAGIC #### `metadata`
# MAGIC
# MAGIC Metadata is meant to be a multi-purposed, unstructured place to put information about a run.
# MAGIC It can be set at the beginning of the run, for example to add custom run-level tags or groupings:
# MAGIC
# MAGIC ```
# MAGIC name: hello-world
# MAGIC image: bash
# MAGIC command: echo 'hello world'
# MAGIC metadata:
# MAGIC   run_type: test
# MAGIC ```
# MAGIC
# MAGIC Metadata on your run is readable through the CLI or SDK:
# MAGIC
# MAGIC ```
# MAGIC > mcli describe run hello-world-VC5nFs
# MAGIC Run Details
# MAGIC Run Name      hello-world-VC5nFs
# MAGIC Image         bash
# MAGIC ...
# MAGIC Run Metadata
# MAGIC KEY         VALUE
# MAGIC run_type    test
# MAGIC ```
# MAGIC
# MAGIC ```
# MAGIC from mcli import get_run
# MAGIC
# MAGIC run = get_run('hello-world-VC5nFs')
# MAGIC print(run.metadata)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model Parameters
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. The Pre-Training YAML for a Llama2 model trained on the c4 dataset

# COMMAND ----------

# MAGIC %%writefile mct_llama2_7b.yaml
# MAGIC # This Llama2-style model for pre-training YAML is organized as follows:
# MAGIC   # Parameters: 
# MAGIC     # Global Seed, Max Sequence Length
# MAGIC     # Model
# MAGIC       # Model Configuration
# MAGIC       # Attention Configuration
# MAGIC       # Feedforward Configuration
# MAGIC     # Tokenizer 
# MAGIC     # Dataloaders
# MAGIC       # Train Loader
# MAGIC       # Eval Loader
# MAGIC     # Optimization
# MAGIC       # Scheduler
# MAGIC       # Optimizer
# MAGIC       # Algorithms
# MAGIC     # System
# MAGIC       # Seed, Device Eval Batch Size, Device Train Microbatch Size, Precision
# MAGIC       # FSDP Configuration
# MAGIC       # Logging
# MAGIC       # Callbacks
# MAGIC         # Speed Monitor
# MAGIC         # LR Monitor
# MAGIC         # Memory Monitor
# MAGIC         # Runtime Estimator
# MAGIC # This section under parameters controls the model and training information
# MAGIC name: llama2-7b-mct
# MAGIC image: mosaicml/llm-foundry:2.3.0_cu121_flash2-latest
# MAGIC
# MAGIC compute:
# MAGIC   gpus: 16
# MAGIC   gpu_type: h100_80gb #h100_80gb # a100_40gb # Replace with the type of GPU in the cluster
# MAGIC   cluster: 'r15z1p3' # 'r7z22p1'  # Replace with the cluster name
# MAGIC
# MAGIC integrations:
# MAGIC   - integration_type: git_repo
# MAGIC     git_branch: v0.8.0 
# MAGIC     git_repo: mosaicml/llm-foundry
# MAGIC     pip_install: .[gpu,databricks] #need databricks for llm-foundry to log checkpoints
# MAGIC     ssh_clone: false
# MAGIC   - integration_type: mlflow
# MAGIC     experiment_name: /Users/sam.raymond@databricks.com/llama2jumpstart/llama2_testing
# MAGIC     tracking_uri: databricks
# MAGIC
# MAGIC command: |-
# MAGIC   cd llm-foundry/scripts
# MAGIC   python data_prep/convert_dataset_hf.py \
# MAGIC     --dataset c4 --data_subset en \
# MAGIC     --out_root ./my-copy-c4 \
# MAGIC     --splits val_small train_small \
# MAGIC     --concat_tokens 2048 --tokenizer meta-llama/Llama-2-7b-hf --eos_text '<|endoftext|>'  
# MAGIC   composer train/train.py /mnt/config/parameters.yaml || (echo "Command failed - killing python" && pkill python && exit 1)
# MAGIC
# MAGIC parameters:
# MAGIC   data_local: ./my-copy-c4
# MAGIC   # data_remote:  # If blank, files must be present in data_local
# MAGIC   max_seq_len: 2048
# MAGIC   global_seed: 17
# MAGIC
# MAGIC   # Run Name
# MAGIC   run_name: llama2_7b # If left blank, will be read from env var $RUN_NAME
# MAGIC
# MAGIC   # Model - https://huggingface.co/meta-llama/Llama-2-7b-hf/blob/main/config.json; https://github.com/meta-llama/llama
# MAGIC   # Mapping challenge: HF config has higher level parameters that are nested, or renamed on Mosaic
# MAGIC   model:
# MAGIC     name: mpt_causal_lm
# MAGIC     init_device: meta
# MAGIC     init_config:
# MAGIC       init_std: 0.02
# MAGIC     d_model: 4096 # hidden_size
# MAGIC     n_heads: 32
# MAGIC     n_layers: 32
# MAGIC     expansion_ratio: 2.6875 # intermediate_size / d_model = 11008 / 4096
# MAGIC     norm_type: rmsnorm # any way to configure the rmsnorm eps value? default = 1e-05 OK in this case since llama uses the same
# MAGIC     learned_pos_emb: false
# MAGIC     max_seq_len: ${max_seq_len}
# MAGIC     vocab_size: 32000
# MAGIC     attn_config:
# MAGIC       attn_impl: flash 
# MAGIC       attn_type: grouped_query_attention #| multihead_attention | multiquery_attention
# MAGIC       attn_uses_sequence_id: false
# MAGIC       kv_n_heads: 32
# MAGIC       clip_qkv: 8
# MAGIC       # rope vs alibi controls whether learning positional embeddings. still possible to use both alibi AND rope. 
# MAGIC       alibi: false # alibi used to shape attention
# MAGIC       rope: true 
# MAGIC       rope_theta: 500000
# MAGIC       
# MAGIC     ffn_config: # config used in MPTBlock, not in modeling_mpt
# MAGIC       ffn_type: mptglu #| mptglu # | mptmlp | te_ln_mlp | mptgeglu
# MAGIC       ffn_act_fn:
# MAGIC         name: silu # | glu | selu
# MAGIC
# MAGIC   # Tokenizer
# MAGIC   tokenizer:
# MAGIC     name: meta-llama/Llama-2-7b-hf
# MAGIC     kwargs:
# MAGIC       model_max_length: ${max_seq_len}
# MAGIC
# MAGIC   # Dataloaders
# MAGIC   train_loader:
# MAGIC     name: text
# MAGIC     dataset:
# MAGIC       local: ${data_local}
# MAGIC       # remote: ${data_remote}
# MAGIC       split: train_small
# MAGIC       shuffle: true
# MAGIC       max_seq_len: ${max_seq_len}
# MAGIC       shuffle_seed: ${global_seed}
# MAGIC     drop_last: true
# MAGIC     num_workers: 8
# MAGIC
# MAGIC   eval_loader:
# MAGIC     name: text
# MAGIC     dataset:
# MAGIC       local: ${data_local}
# MAGIC       # remote: ${data_remote}
# MAGIC       split: val_small
# MAGIC       shuffle: false
# MAGIC       max_seq_len: ${max_seq_len}
# MAGIC       shuffle_seed: ${global_seed}
# MAGIC     drop_last: false
# MAGIC     num_workers: 8
# MAGIC
# MAGIC   # Optimization
# MAGIC   scheduler:
# MAGIC     name: cosine_with_warmup
# MAGIC     t_warmup: 100ba
# MAGIC     alpha_f: 0.1
# MAGIC
# MAGIC   optimizer:
# MAGIC     name: decoupled_adamw
# MAGIC     lr: 3.0e-4
# MAGIC     betas:
# MAGIC     - 0.9
# MAGIC     - 0.95
# MAGIC     eps: 1.0e-08
# MAGIC     weight_decay: 0.000012
# MAGIC
# MAGIC   algorithms:
# MAGIC     gradient_clipping:
# MAGIC       clipping_type: norm
# MAGIC       clipping_threshold: 1.0
# MAGIC
# MAGIC   max_duration: 500ba #63900ba  # ~ 134B tokens
# MAGIC   eval_interval: 200ba
# MAGIC   eval_first: false
# MAGIC   eval_subset_num_batches: -1
# MAGIC   global_train_batch_size: 1024 # | 1024
# MAGIC
# MAGIC   # System
# MAGIC   seed: ${global_seed}
# MAGIC   device_eval_microbatch_size: auto
# MAGIC   device_train_microbatch_size: auto # | auto 
# MAGIC   precision:  amp_bf16 # | fp32 # | amp_fp16 # | amp_bf16
# MAGIC
# MAGIC   # FSDP
# MAGIC   fsdp_config:
# MAGIC     sharding_strategy: HYBRID_SHARD # | HYBRID_SHARD
# MAGIC     mixed_precision: PURE # FULL | PURE | DEFAULT
# MAGIC     activation_checkpointing: false #true
# MAGIC     activation_checkpointing_reentrant: false
# MAGIC     activation_cpu_offload: false
# MAGIC     limit_all_gathers: false
# MAGIC     state_dict_type: sharded # | sharded | local | full
# MAGIC     backward_prefetch: BACKWARD_POST
# MAGIC
# MAGIC   # Garbage Collection
# MAGIC   scheduled_gc:
# MAGIC     batch_interval: 2000
# MAGIC
# MAGIC   # Logging
# MAGIC   progress_bar: false
# MAGIC   log_to_console: true
# MAGIC   console_log_interval: 100ba
# MAGIC
# MAGIC   callbacks:
# MAGIC     hf_checkpointer: ## This is needed to log to Databricks UC along witht he mlflow logger
# MAGIC       overwrite: true
# MAGIC       precision: bfloat16
# MAGIC       save_folder: dbfs:/databricks/mlflow-tracking/{mlflow_experiment_id}/{mlflow_run_id}/artifacts/checkpoints
# MAGIC       save_interval: 100ba
# MAGIC       mlflow_logging_config:
# MAGIC         task: llm/v1/chat
# MAGIC         metadata:
# MAGIC           task: llm/v1/chat
# MAGIC       mlflow_registered_model_name: llama2_mct
# MAGIC     speed_monitor:
# MAGIC       window_size: 10
# MAGIC     lr_monitor: {}
# MAGIC     memory_monitor: {}
# MAGIC     runtime_estimator: {}
# MAGIC
# MAGIC   loggers:
# MAGIC     mlflow:
# MAGIC       experiment_name: /Users/sam.raymond@databricks.com/llama2_testing
# MAGIC       tracking_uri: databricks
# MAGIC       model_registry_uri: databricks-uc #databricks | databricks-uc
# MAGIC       model_registry_prefix: sam_raymond_ti3w_da.dbacademy_adv_genai_course
# MAGIC       
# MAGIC
# MAGIC   # Checkpoint to local filesystem or remote object store
# MAGIC   save_interval: 500ba
# MAGIC   save_num_checkpoints_to_keep: 1  # Important, this cleans up checkpoints saved to DISK
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Common MCLI Commands
# MAGIC
# MAGIC | Command                                         | Description                                                                                   |
# MAGIC |-------------------------------------------------|-----------------------------------------------------------------------------------------------|
# MAGIC | `mcli run -f <your_yaml>`                       | Submit a run with the provided YAML configuration.                                            |
# MAGIC | `mcli get runs`                                 | List all of your submitted runs. Use `mcli get runs --help` to view the many filters available. |
# MAGIC | `mcli describe run <run_name>`                  | Get detailed information about a run, including the config that was used to launch it.       |
# MAGIC | `mcli logs <run_name>`                          | Retrieve the console log of the latest resumption of the indicated run.                      |
# MAGIC | `mcli stop run <run_name>`                      | Stop the provided run. The run will be stopped but not deleted from the cluster.             |
# MAGIC | `mcli run -r <stopped_run>`                     | Restart a stopped run.                                                                       |
# MAGIC | `mcli delete run <run_name>`                    | Delete the run (and its associated logs) from the cluster. We recommend using this sparingly.|
# MAGIC | `mcli update run <run_name> --max-duration <hours>` | Update a handful of run parameters, like the max time (in hours) that a run can run for. |
# MAGIC
# MAGIC Additional `update run` usage:
# MAGIC
# MAGIC ```bash
# MAGIC mcli update run [-h] [--priority PRIORITY] [--no-preemptible | --preemptible] [--max-retries MAX_RETRIES] [--max-duration MAX_DURATION] <run_name>
# MAGIC ```
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### MosaicML Secrets Management
# MAGIC
# MAGIC Like Databricks, MosaicML allows for the creation of secrets to avoid any plain text api keys or access tokens. A [full list of supported secret types](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/index.html#supported-secret-types) are show in the secrets documentation. 
# MAGIC
# MAGIC We will use secrets to first store the MosaicML API key that we need to send commands to the MosaicML platform to run our code. The first step is to connect to MosaicML:
# MAGIC
# MAGIC 1. Create API keys/tokens from MosaicML
# MAGIC 1. Use Databricks Secrets to store these sensitive keys to the workspace you are using this notebook from. 
# MAGIC
# MAGIC You can find or create a MosaicML API key by visiting: https://console.mosaicml.com/account and signing into your organization. Once this is done, setup a Databricks secret in the workspace.
# MAGIC
# MAGIC <img src="https://imagizer.imageshack.com/img922/760/UvGSax.png" width="1000">

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Building a Pre-Training Run

# COMMAND ----------

# MAGIC %md
# MAGIC ### MCT Pre-Training Preparation

# COMMAND ----------

# MAGIC %pip install mosaicml-cli

# COMMAND ----------

MOSAIC_API_KEY        = dbutils.secrets.get(scope='gen_ai',key='mosaic_api_key') 
HF_TOKEN              = dbutils.secrets.get(scope='gen_ai',key='llama2_hftoken')
DB_HOST               = dbutils.secrets.get(scope="gen_ai",key="databricks_workspace_url")
DB_TOKEN              = dbutils.secrets.get(scope="gen_ai",key="databricks_workspace_token")

# COMMAND ----------

import os
import mcli 
import pandas as pd
from mcli.objects.secrets import SecretType, MCLIDatabricksSecret
from mcli.objects.secrets import SecretType, MCLIEnvVarSecret
from mcli.objects.secrets import SecretType, MCLIS3Secret

# COMMAND ----------

# Firstly, we need to set our API key. 
mcli.set_api_key(MOSAIC_API_KEY)

# COMMAND ----------

# mcli.delete_secrets()
mcli.get_secrets()

# COMMAND ----------

# Firstly we need to add our MosaicML API token to run on the platform. 

mcli_api_secret = MCLIEnvVarSecret(
    name="mosaic-api-key", # This can be any string you want
    secret_type=SecretType.environment,
    key="MOSAICML_API_KEY", # Do not change this key
    value = MOSAIC_API_KEY,
)
mcli.create_secret(mcli_api_secret)

# COMMAND ----------

hfhub_api_secret = MCLIEnvVarSecret(
    name="hf-api-key", # This can be any string you want
    secret_type=SecretType.environment,
    key="HUGGING_FACE_HUB_TOKEN", # Do not change this key
    value = HF_TOKEN
)
mcli.create_secret(hfhub_api_secret)

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLflow Integration  
# MAGIC You can track the progress of your runs using MLflow in Databricks. For this you will need to create a [databricks secret](https://docs.mosaicml.com/projects/mcli/en/latest/resources/secrets/databricks.html) which consists of:
# MAGIC - The url of the workspace you are using
# MAGIC - A PAT token to authorize access to the workspace from MosaicML to the workspace for MLflow
# MAGIC
# MAGIC To create the secret, use a third field, `--name` to describe the token, here `databricks_mlflow`
# MAGIC
# MAGIC Finally in the run configuration, you will need to specify the experiment tracker as follows:
# MAGIC ```python
# MAGIC { 
# MAGIC   'integration_type': 'mlflow',
# MAGIC   'experiment_name': '/Users/<user_name>/<experiment_name>', #An experiment name must be an absolute path within the  Databricks workspace, e.g. '/Users/<some-username>/my-experiment'.
# MAGIC }
# MAGIC ```

# COMMAND ----------

# For MLflow integration, you will need a token for the workspace you are running in. Create a secret for the token, and the workspace url. 

databricks_secret = MCLIDatabricksSecret(
    name="databricks_mlflow", # This can be any string you want
    secret_type=SecretType.databricks,
    host=DB_HOST,
    token=DB_TOKEN,
)
mcli.create_secret(databricks_secret)

# COMMAND ----------

def view_cluster_info(data):
    # Extracting relevant information and creating a dataframe
    rows = []
    for cluster in data:
        for instance in cluster.cluster_instances:
            rows.append({
                'Cluster Name': cluster.name,
                'Provider': cluster.provider,
                'Allow Fractional': cluster.allow_fractional,
                'Allow Multinode': cluster.allow_multinode,
                'Instance Name': instance.name,
                'GPU Type': instance.gpu_type,
                'GPUs': instance.gpus,
                'CPUs': instance.cpus,
                'Memory': instance.memory,
                'Storage': instance.storage,
                'Nodes': instance.nodes
            })
    df = pd.DataFrame(rows)
    return df

view_cluster_info(mcli.get_clusters())

# COMMAND ----------

# This helper function will let us see the latest logs for each of our runs.
def show_latest_log(run):
    print(f"{list(mcli.get_run_logs(run=run.name))[-1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Launching the training run

# COMMAND ----------

!mcli run -f ./mct_llama2_7b.yaml --priority low --preemptible

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Monitoring Training Run Performance

# COMMAND ----------

!mcli util

# COMMAND ----------

# !mcli get run <run_name_here>

# COMMAND ----------

# !mcli logs <run_name_here>

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
