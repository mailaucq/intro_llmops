# Databricks notebook source
# MAGIC %md
# MAGIC # 3. Comparar y decidir (gate)
# MAGIC
# MAGIC Lee las métricas de las dos tasks anteriores vía `taskValues`, compara
# MAGIC el candidato contra los thresholds, y si pasa mueve el alias
# MAGIC `@production`. El resultado (`gate_passed`) lo lee la condition task
# MAGIC `gate_check` para decidir si el pipeline sigue hacia registrar y
# MAGIC desplegar el modelo, o se detiene aquí.
# MAGIC
# MAGIC No instala nada ni reinicia Python: solo lee/escribe métricas y mueve
# MAGIC un alias, no necesita el cliente de OpenAI.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "llmops")
dbutils.widgets.text(
    "thresholds",
    '{"correctness": 0.8, "safety": 1.0, "answers_directly": 0.8, "has_code_block": 0.7}',
)

# COMMAND ----------

import json

import mlflow.genai

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
THRESHOLDS = json.loads(dbutils.widgets.get("thresholds"))
PROMPT_NAME = f"{CATALOG}.{SCHEMA}.python_qa"

# COMMAND ----------

metrics_v1 = dbutils.jobs.taskValues.get(taskKey="eval_baseline", key="metrics", default={}, debugValue={})
version_v1 = dbutils.jobs.taskValues.get(taskKey="eval_baseline", key="prompt_version", default=None, debugValue=1)
metrics_v2 = dbutils.jobs.taskValues.get(taskKey="eval_candidate", key="metrics", default={}, debugValue={})
version_v2 = dbutils.jobs.taskValues.get(taskKey="eval_candidate", key="prompt_version", default=None, debugValue=2)

print("v1:", version_v1, metrics_v1)
print("v2:", version_v2, metrics_v2)

# COMMAND ----------

failed = [
    name for name, min_value in THRESHOLDS.items()
    if metrics_v2.get(name, 0.0) < min_value
]
gate_passed = len(failed) == 0

if gate_passed:
    print("Gate: PASA — el candidato cumple todos los thresholds")
else:
    print(f"Gate: FALLA — no cumple: {failed}")

winner_version = version_v2 if gate_passed else version_v1

# COMMAND ----------

if gate_passed:
    mlflow.genai.set_prompt_alias(name=PROMPT_NAME, alias="production", version=winner_version)
    print(f"{PROMPT_NAME}@production -> v{winner_version}")
else:
    print("No se promueve: el candidato no pasó el gate. @production no se toca.")

# COMMAND ----------

dbutils.jobs.taskValues.set("gate_passed", "true" if gate_passed else "false")
dbutils.jobs.taskValues.set("winner_version", winner_version)
