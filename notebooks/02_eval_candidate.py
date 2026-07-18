# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Candidato — prompt v2
# MAGIC
# MAGIC Hipótesis: el prompt genérico produce preámbulo y respuestas largas sin
# MAGIC código. Lo hacemos explícito, movemos el alias `champion` a esta versión,
# MAGIC y evaluamos. `python_qa` no cambia — solo a qué versión apunta el alias.

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

v2 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=[
        {"role": "system", "content":
            "You are an expert Python engineer answering questions from other developers.\n"
            "\n"
            "Rules:\n"
            "- Answer directly. No preamble, no restating the question.\n"
            "- Always include a minimal, runnable code example in a ```python block.\n"
            "- Keep it under 150 words.\n"
            "- If the question is ambiguous, answer the most common interpretation.\n"
            "- If you are unsure, say so rather than inventing an API."},
        {"role": "user", "content": "{{question}}"},
    ],
    commit_message="v2: rol explícito, prohibido preámbulo, exige bloque de código, tope de 150 palabras",
)
mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="champion", version=v2.version)
print("prompt v2 version:", v2.version)

# COMMAND ----------

with mlflow.start_run(run_name="prompt_v2") as run_v2:
    results_v2 = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=python_qa,
        scorers=scorers,
    )

metrics_v2 = tidy_metrics(results_v2.metrics)
print(metrics_v2)

# COMMAND ----------

dbutils.jobs.taskValues.set("prompt_version", v2.version)
dbutils.jobs.taskValues.set("metrics", metrics_v2)
dbutils.jobs.taskValues.set("run_id", run_v2.info.run_id)
