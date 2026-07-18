# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Baseline — prompt v1
# MAGIC
# MAGIC Registra el prompt genérico, lo evalúa, y deja sus métricas y versión
# MAGIC disponibles para la siguiente task vía `dbutils.jobs.taskValues`.

# COMMAND ----------

# MAGIC %pip install -q --upgrade "mlflow[databricks]>=3.1.0" openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "llmops")
dbutils.widgets.text("source_jsonl", "")
dbutils.widgets.text("model_endpoint", "databricks-qwen35-122b-a10b")

# COMMAND ----------

# MAGIC %run ./00_common

# COMMAND ----------

v1 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=[
        {"role": "system", "content":
            "You are a helpful assistant that answers Python programming questions."},
        {"role": "user", "content": "{{question}}"},
    ],
    commit_message="v1: baseline, prompt genérico del lab",
)
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="champion", version=v1.version)
print("prompt v1 version:", v1.version)

# COMMAND ----------

with mlflow.start_run(run_name="prompt_v1") as run_v1:
    results_v1 = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=python_qa,
        scorers=scorers,
    )

metrics_v1 = tidy_metrics(results_v1.metrics)
print(metrics_v1)

# COMMAND ----------

dbutils.jobs.taskValues.set("prompt_version", v1.version)
dbutils.jobs.taskValues.set("metrics", metrics_v1)
dbutils.jobs.taskValues.set("run_id", run_v1.info.run_id)
