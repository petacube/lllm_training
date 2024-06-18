# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab: Continued Pre-Training and Instruction Fine-Tuning
# MAGIC
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you will:<br>
# MAGIC - Build a new, custom model to generate blog article titles from article text, in the style of Databricks. 
# MAGIC - Create CPT and IFT datasets
# MAGIC - Fine-tune Llama3-8B with CPT and IFT using the FM API
# MAGIC - Serve the trained model and compare with an off-the-shelf model

# COMMAND ----------

# MAGIC %pip install --quiet databricks-genai==1.0.2 openai==1.30.3
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

import os, json, openai
import pandas as pd
import pyspark.sql.functions as F

from databricks.model_training import foundation_model as ft
from mlflow.utils.databricks_utils import get_databricks_env_vars
from openai import OpenAI
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql import DataFrame
from typing import Iterator, List

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Setting up our CPT data
# MAGIC
# MAGIC Currently the data we have access to are the downloaded html pages from the blogs website. We'll need to extract from this the headline and the article text, so that we can then write out each blog as a `.txt` file for continued pre-training.
# MAGIC

# COMMAND ----------

# Let's create a new Volume of our own to convert this to .txt format so we can do CPT

# Create a volume to store raw text files for CPT
VOLUME_CPT = 'lab_raw_text'
_ = spark.sql(f'CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME_CPT}')

blog_df = spark.read.table(f"{CATALOG}.{SCHEMA}.blogs_bronze")

# COMMAND ----------

# Specify the output directory
output_dir = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_CPT}"

# Define a UDF to write text to files
def write_text_file(id, text):
    file_path = os.path.join(output_dir, f"{id}.txt")
    with open(file_path, "w") as file:
        file.write(text)
    return file_path

# Register the UDF
write_text_file_udf = udf(write_text_file, StringType())

# Apply the UDF to the DataFrame and ensure all rows are processed
df_with_path = blog_df.withColumn("file_path", write_text_file_udf("id", "text"))

# Use a transformation and action to process all rows
df_with_path.select("file_path").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Setting up our IFT data
# MAGIC
# MAGIC Our IFT data is already set up for us in a UC Volume. Inside are two files, one for training, one for evaluation. We will create a new Volume and copy these two files over.

# COMMAND ----------

# Let's create another new Volume of our own to copy this training data over for our IFT 
 
# Create a volume to store raw text files for IFT
VOLUME_IFT = 'lab_ift_data'
_ = spark.sql(f'CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME_IFT}')

# COMMAND ----------

def load_and_filter(table_name: str, response_col: str = "title") -> DataFrame:
    """
    Load table and filter null or empty strings in 'text' or `response_col`.

    Args:
        table_name: The name of the table to load.
        response_col: The column to filter for null or empty strings.

    Returns:
        Filtered DataFrame.
    """
    print(f"Loading table: {table_name}")
    df = spark.table(table_name)
    original_count = df.count()
    print(f"Row count: {original_count}")

    print(f"\nFilter null or empty strings in 'text' or '{response_col}'")
    filtered_df = filter_null_or_empty(df, ["text", response_col])
    filtered_count = filtered_df.count()
    print(f"Number of rows dropped: {original_count - filtered_count}")
    print(f"Filtered count: {filtered_count}")

    return filtered_df
  

def filter_null_or_empty(df: DataFrame, columns: List[str]) -> DataFrame:
    """
    Filter rows where any of the specified columns are null or empty.

    Args:
        df: The DataFrame to filter.
        columns: The list of columns to check for null or empty values.

    Returns:
        Filtered DataFrame.
    """
    print("Filter rows where any of the specified columns are null or empty...")
    for col in columns:
        print(f"\tColumn: {col}")
        df = df.filter((F.col(col).isNotNull()) & (F.col(col) != ""))
    return df

filtered_df = load_and_filter(table_name=f"{CATALOG}.{SCHEMA}.blogs_bronze")  
filtered_deduped_df = filtered_df.drop_duplicates(subset=["text", "title"])
filtered_deduped_count = filtered_deduped_df.count()
print(f"Final deduplicated count: {filtered_deduped_count}")

# COMMAND ----------

class PromptTemplate:
    """Class to represent a prompt template for instruction dataset generation."""

    def __init__(self, instruction: str, blog_key: str, response_key: str) -> None:
        self.instruction = instruction
        self.blog_key = blog_key
        self.response_key = response_key

    def generate_prompt(self, blog_text: str) -> str:
        """
        Generate a prompt using the template and the given blog text.

        Args:
            blog_text: The text of the blog.

        Returns:
            Prompt template.
        """
        return f"""{self.instruction}
{self.blog_key}
{blog_text}
{self.response_key}
"""

blog_title_generation_template = PromptTemplate(
    instruction="The following is the text of a Databricks blog post. Create a title for the provided blog post.",
    blog_key="### Blog:",
    response_key="### Title:"
)  

def add_instruction_prompt_column(df: DataFrame, prompt_template: PromptTemplate) -> DataFrame:
    """
    Add 'prompt' column to the DataFrame using the specified template.

    Args:
        df: Input DataFrame.
        prompt_template: Prompt template to use for generating prompts.

    Returns:
        DataFrame with 'prompt' column.
    """
    @F.pandas_udf(StringType())
    def generate_prompt(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        for texts in batch_iter:
            prompts = texts.apply(prompt_template.generate_prompt)
            yield prompts

    return df.withColumn("prompt", generate_prompt(df["text"]))

# Add prompt column 
instruction_df = add_instruction_prompt_column(filtered_deduped_df, blog_title_generation_template)

# subset to prompt col, and rename title to response
instruction_df = instruction_df.selectExpr("prompt", "title as response")

# Define a UDF to format each row as a message for our Chat Completion model
@F.udf("string")
def format_as_message(prompt, response):
    return json.dumps({
        "messages": [
            {"role": "system", "content": "A conversation between a user and a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
    })

# Apply the UDF to the DataFrame
formatted_df = instruction_df.withColumn("message", format_as_message(F.col("prompt"), F.col("response"))).drop("prompt","response")
display(formatted_df)

# COMMAND ----------

# Create and write out train/eval IFT data

instruction_df_train, instruction_df_eval = formatted_df.randomSplit([0.9,0.1], seed=42)

def write_df_to_jsonl(df, path):
    temp_path = "/tmp/temp_json"
    df.select("message").coalesce(1).write.mode("overwrite").text(temp_path)

    part_files = dbutils.fs.ls(temp_path)
    json_part_file = next((file.path for file in part_files if file.name.startswith("part-")), None)
    
    if json_part_file:
        final_path = path
        dbutils.fs.mv(json_part_file, final_path)
        dbutils.fs.rm(temp_path, recurse=True)
        print(f"Successfully wrote Spark DataFrame as JSONL to {final_path}")
    else:
        print("No part file found. Check the temp path.")

# Write training and evaluation DataFrames to JSONL files
train_ift_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_IFT}/train.jsonl"
eval_ift_path = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_IFT}/eval.jsonl"

write_df_to_jsonl(instruction_df_train, train_ift_path)
write_df_to_jsonl(instruction_df_eval, eval_ift_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Fine-tuning API
# MAGIC
# MAGIC Now that we have our data sorted out, we are ready to build our custom model. We will use the new Databricks Fine-tuning API for both the Continued Pre-Training and Instruction Fine-tuning.
# MAGIC
# MAGIC #### Databricks Generative AI Fine-tuning API
# MAGIC
# MAGIC In many cases, you may want to use models other than those in the [Foundation API](https://docs.databricks.com/en/machine-learning/foundation-models/index.html). To facilitate this, we will use the new Databricks Mosaic AI finetuning API. This API runs within Databricks and allows you to continuously pre-train a model, as well as instruction finetune it. 
# MAGIC
# MAGIC In this notebook we will cover the important components to configure finetuning runs and work with data in the Unity Catalog. 
# MAGIC
# MAGIC
# MAGIC **As a reminder** 
# MAGIC - For Continued Pre-Training, the data is required to be a UC volume with a folder of *.txt files
# MAGIC - For Instruction Fine-Tuning, the data is required to be a UC volume with a *.jsonl file that contains a `messages` array with `"role":"user","content","..." and "role":"assistant","content":"..."` pairs.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3a: Continued Pre-training
# MAGIC
# MAGIC In this part of Step 3 we will take a base model, in this case Llama3-8B, and perform CPT on the text data we built in Step 1.

# COMMAND ----------

# Create the CPT fine-tuning run for meta-llama/Meta-Llama-3-8B

user = DA.catalog_name_prefix
exp_name = "Llama3_CPT"

continued_pretraining_run = ft.create(
    model="<FILL_IN>",
    train_data_path="<FILL_IN>", # Hint this will be of the form: "dbfs:_Path_to_your_cpt_files" 
    register_to="<FILL_IN>", # Register this to your UC Volume
    experiment_path=f"/Users/{DA.username}/{exp_name}",
    task_type="<FILL_IN>", 
    training_duration="<FILL_IN>", # Hint: Set this to 10 Million tokens
    )
print(f"Finetuning run: {continued_pretraining_run.name} sent to compute cluster")

# COMMAND ----------

# We can track the stages of the run using ft.get_events(fintuning_run.name)
ft.get_events(continued_pretraining_run.name)

# COMMAND ----------

# To see more information of the run we created, we can use the ft.get() command
cpt_info = ft.get(continued_pretraining_run.name)
print(f"Status: {cpt_info.status}\nDetails: {cpt_info.details}")
print(cpt_info)

# COMMAND ----------

# MAGIC %md
# MAGIC **NOTE**:
# MAGIC We will need to wait for this run to complete before using the final checkpoint in the IFT section below

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3b: Instruction Fine-Tuning (IFT)
# MAGIC
# MAGIC For IFT we can either use a base model, where the weights are loaded in from a public repo, or we can use the checkpoint of the model we pretrained. To get the custom weights, we need to get the checkpoint file from the finished CPT run. It will be in a location that looks like:
# MAGIC `checkpoint = "dbfs:/databricks/mlflow-tracking/<experiment_id>/<run_id>/artifacts/<path>"`
# MAGIC You can get this path from the mlflow experiment UI.

# COMMAND ----------

# MAGIC %md
# MAGIC **Example Training Data Sample**
# MAGIC
# MAGIC Below is an example of the training/evaluation data for the instruction fine-tuning:
# MAGIC ```
# MAGIC {"prompt":"The following is the text of a Databricks blog post. Create a title for the provided blog post.\n### Blog:\nAs part of MLflow 2’s support for LLMOps, we are excited to introduce the latest updates to support prompt engineering in MLflow 2.7. Assess LLM project viability with an interactive prompt interface Prompt engineering is a great way to quickly assess if a use case can be solved with a large language model (LLM). With the new prompt engineering UI in MLflow 2.7, business stakeholders can experiment with various base models, parameters, and prompts to see if outputs are promising enough to start a new project. Simply create a new Blank Experiment or (open an existing one) and click “New Run” to access the interactive prompt engineering tool. Sign up to join the preview here. Automatically track prompt engineering experiments to build evaluation datasets and identify best model candidates In the new prompt engineering UI, users can explicitly track an evaluation run by clicking “Create run” to log results to MLflow. This button tracks the set of parameters, base models, and prompts as an MLflow model and outputs are stored in an evaluation table. This table can then be used for manual evaluation, converted to a Delta table for deeper analysis in SQL, or be used as the test dataset in a CI/CD process. MLflow is always adding more metrics in the MLflow Evaluation API to help you identify the best model candidate for production, now including toxicity and perplexity. Users can use the MLflow Table or Chart view to compare model performances: Because the set of parameters, prompts, and base models are logged as MLflow models, this means you can deploy a fixed prompt template for a base model with set parameters in batch inference or serve it as an API with Databricks Model Serving. For LangChain users this is especially useful as MLflow comes with model versioning. Democratize ad-hoc experimentation across your organization with guardrails The MLflow prompt engineering UI works with any MLflow AI Gateway route. AI Gateway routes allow your organization to centralize governance and policies for SaaS LLMs; for example, you can put OpenAI’s GPT-3.5-turbo behind a Gateway route that manages which users can query the route, provides secure credential management, and provides rate limits. This protects against abuse and gives platform teams confidence to democratize access to LLMs across their org for experimentation. The MLflow AI Gateway supports OpenAI, Cohere, Anthropic, and Databricks Model Serving endpoints. However, with generalized open source LLMs getting more and more competitive with proprietary generalized LLMs, your organization may want to quickly evaluate and experiment with these open source models. You can now also call MosaicML’s hosted Llama2-70b-chat. Try MLflow today for your LLM development! We are working quickly to support and standardize the most common workflows for LLM development in MLflow. Check out this demo notebook to see how to use MLflow for your use cases. For more resources: Sign up for the MLflow AI Gateway Preview (includes the prompt engineering UI) here. To get started with the prompt engineering UI, simply upgrade your MLflow version (pip install –upgrade mlflow), create an MLflow Experiment, and click “New Run”. To evaluate various models on the same set of questions, use the MLflow Evaluation API. If there is a SaaS LLM endpoint you want to support in the MLflow AI Gateway, follow the guidelines for contribution on the MLflow repository. We love contributions!\n### Title:\n",
# MAGIC
# MAGIC "response":
# MAGIC "Introducing MLflow 2.7 with new LLMOps capabilities"
# MAGIC }
# MAGIC ```

# COMMAND ----------

user = DA.catalog_name_prefix
exp_name = "Llama3_IFT"
custom_weights_path = f"{cpt_info.save_folder}/{cpt_info.name}/checkpoints/latest-sharded-rank0.symlink"

if cpt_info.status.value == "COMPLETED":
    print(f"Checkpoint available at {custom_weights_path}")
else:
    raise Exception(
        f"No model checkpoint available. Please wait for the CPT fine-tuning run to complete before continuing.")

instruction_finetuning_run = ft.create(
    model="<FILL_IN>",
    train_data_path="<FILL_IN>", # Hint this will be of the form: "dbfs:_Path_to_your_cpt_files" 
    eval_data_path="<FILL_IN>",
    register_to="<FILL_IN>",  # Register this to your UC Volume
    experiment_path=f"/Users/{DA.username}/{exp_name}", 
    task_type="<FILL_IN>",
    training_duration="1000000tok", 
    custom_weights_path="<FILL_IN>", # Use the custom weights path from the CPT run provided by the instructor
    )

print(f"Finetuning run: {instruction_finetuning_run.name} sent to compute cluster")

# COMMAND ----------

# We can track the stages of the run using ft.get_events(fintuning_run.name)
ft.get_events(instruction_finetuning_run.name)

# COMMAND ----------

# To see more information of the run we created, we can use the ft.get() command
info = ft.get(instruction_finetuning_run.name)
print(f"Status: {info.status}\nDetails: {info.details}")
print(info)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Serving our trained Model

# COMMAND ----------

# For this session, use the provided endpoint: "adv_genai_cpt_ift_model_lab"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Model Inference
# MAGIC
# MAGIC Now that you have trained a custom model with CPT and then used that checkpoint for the IFT training. Next we can test how well our served model performs the task of generating headlines, compared to the general Llama3-8B model, hosted on the Foundation Model API.

# COMMAND ----------

def get_api_credentials():
    mlflow_db_creds = get_databricks_env_vars("databricks")
    api_token = mlflow_db_creds["DATABRICKS_TOKEN"]
    api_root = mlflow_db_creds["_DATABRICKS_WORKSPACE_HOST"]
    return api_token, api_root

token,root = get_api_credentials()
client = OpenAI(
    api_key=token,
    base_url=root+"/serving-endpoints"
)

# COMMAND ----------

user_message=[
    {"role":"user",
     "content":"The following is the text of a Databricks blog post. Create a title for the provided blog post.\n### Blog:\nAs part of MLflow 2’s support for LLMOps, we are excited to introduce the latest updates to support prompt engineering in MLflow 2.7. Assess LLM project viability with an interactive prompt interface Prompt engineering is a great way to quickly assess if a use case can be solved with a large language model (LLM). With the new prompt engineering UI in MLflow 2.7, business stakeholders can experiment with various base models, parameters, and prompts to see if outputs are promising enough to start a new project. Simply create a new Blank Experiment or (open an existing one) and click “New Run” to access the interactive prompt engineering tool. Sign up to join the preview here. Automatically track prompt engineering experiments to build evaluation datasets and identify best model candidates In the new prompt engineering UI, users can explicitly track an evaluation run by clicking “Create run” to log results to MLflow. This button tracks the set of parameters, base models, and prompts as an MLflow model and outputs are stored in an evaluation table. This table can then be used for manual evaluation, converted to a Delta table for deeper analysis in SQL, or be used as the test dataset in a CI/CD process. MLflow is always adding more metrics in the MLflow Evaluation API to help you identify the best model candidate for production, now including toxicity and perplexity. Users can use the MLflow Table or Chart view to compare model performances: Because the set of parameters, prompts, and base models are logged as MLflow models, this means you can deploy a fixed prompt template for a base model with set parameters in batch inference or serve it as an API with Databricks Model Serving. For LangChain users this is especially useful as MLflow comes with model versioning. Democratize ad-hoc experimentation across your organization with guardrails The MLflow prompt engineering UI works with any MLflow AI Gateway route. AI Gateway routes allow your organization to centralize governance and policies for SaaS LLMs; for example, you can put OpenAI’s GPT-3.5-turbo behind a Gateway route that manages which users can query the route, provides secure credential management, and provides rate limits. This protects against abuse and gives platform teams confidence to democratize access to LLMs across their org for experimentation. The MLflow AI Gateway supports OpenAI, Cohere, Anthropic, and Databricks Model Serving endpoints. However, with generalized open source LLMs getting more and more competitive with proprietary generalized LLMs, your organization may want to quickly evaluate and experiment with these open source models. You can now also call MosaicML’s hosted Llama2-70b-chat. Try MLflow today for your LLM development! We are working quickly to support and standardize the most common workflows for LLM development in MLflow. Check out this demo notebook to see how to use MLflow for your use cases. For more resources: Sign up for the MLflow AI Gateway Preview (includes the prompt engineering UI) here. To get started with the prompt engineering UI, simply upgrade your MLflow version (pip install –upgrade mlflow), create an MLflow Experiment, and click “New Run”. To evaluate various models on the same set of questions, use the MLflow Evaluation API. If there is a SaaS LLM endpoint you want to support in the MLflow AI Gateway, follow the guidelines for contribution on the MLflow repository. We love contributions!"}
    ]

# COMMAND ----------

# Generate responses from both the model you've built, and the Llama-3-70b-Instruct model, using the openai client initialized above. 

chat_completion_FMAPI = client.chat.completions.create(
  messages="<FILL_IN>",
  model="<FILL_IN>",
  max_tokens=256
)       

chat_completion_Finetuned = client.chat.completions.create(
  messages="<FILL_IN>",
  model="adv_genai_cpt_ift_model_lab", # Use the pre fine-tuned model that your instructor has made available
  max_tokens=256
)

# COMMAND ----------

print(chat_completion_FMAPI.choices[0].message.content)

# COMMAND ----------

print(chat_completion_Finetuned.choices[0].message.content)

# COMMAND ----------

# MAGIC %md
# MAGIC **CONGRATULATIONS!** 
# MAGIC You've now performed domain tuning via CPT and task specialization with IFT using the Foundation Model Training API!

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>
