# Databricks notebook source
# MAGIC %md
# MAGIC # 5. Desplegar — Model Serving endpoint
# MAGIC
# MAGIC Crea el endpoint si no existe, o actualiza su config para servir la
# MAGIC versión recién registrada del `ResponsesAgent`. Este es el paso que
# MAGIC cierra el loop: el prompt versionado ya vive detrás de un endpoint
# MAGIC HTTP real, no solo en el registry.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "llmops")
dbutils.widgets.text("endpoint_name", "qa-model-serving")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.sdk.errors import ResourceDoesNotExist

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.qa_model"

MODEL_VERSION = dbutils.jobs.taskValues.get(
    taskKey="register_model", key="registered_model_version", default=None, debugValue="1"
)

w = WorkspaceClient()

served_entities = [
    ServedEntityInput(
        entity_name=UC_MODEL_NAME,
        entity_version=str(MODEL_VERSION),
        workload_size="Small",
        scale_to_zero_enabled=True,
    )
]

# COMMAND ----------

try:
    w.serving_endpoints.get(ENDPOINT_NAME)
    exists = True
except ResourceDoesNotExist:
    exists = False

if exists:
    print(f"Actualizando {ENDPOINT_NAME} -> {UC_MODEL_NAME} v{MODEL_VERSION}")
    w.serving_endpoints.update_config_and_wait(name=ENDPOINT_NAME, served_entities=served_entities)
else:
    print(f"Creando {ENDPOINT_NAME} -> {UC_MODEL_NAME} v{MODEL_VERSION}")
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(served_entities=served_entities),
    )

print(f"Endpoint listo: {ENDPOINT_NAME}")
