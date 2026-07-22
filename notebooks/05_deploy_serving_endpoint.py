# Databricks notebook source
# MAGIC %md
# MAGIC # 5. Desplegar — Model Serving endpoint
# MAGIC
# MAGIC Crea el endpoint si no existe, o actualiza su config para servir la
# MAGIC versión recién registrada del `ResponsesAgent`. Este es el paso que
# MAGIC cierra el loop: el prompt versionado ya vive detrás de un endpoint
# MAGIC HTTP real, no solo en el registry.

# COMMAND ----------

# MAGIC %pip install -q --upgrade "mlflow[databricks]>=3.1.0" openai databricks-sdk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------


dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "llmops")
dbutils.widgets.text("endpoint_name", "qa-model-serving")
dbutils.widgets.text("rate_limit_per_minute", "20")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import (
    AiGatewayGuardrailParameters,
    AiGatewayGuardrailPiiBehavior,
    AiGatewayGuardrailPiiBehaviorBehavior,
    AiGatewayGuardrails,
    AiGatewayRateLimit,
    AiGatewayRateLimitKey,
    AiGatewayRateLimitRenewalPeriod,
    AiGatewayUsageTrackingConfig,
    EndpointCoreConfigInput,
    ServedEntityInput,
)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")
RATE_LIMIT_PER_MINUTE = int(dbutils.widgets.get("rate_limit_per_minute"))
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

# MAGIC %md ### AI Gateway — guardrails, rate limit, usage tracking
# MAGIC
# MAGIC - **Guardrails**: filtro en tiempo real, aparte del scorer `Safety` del eval
# MAGIC   (ese solo mide en el momento de evaluar, no protege tráfico real). `pii`
# MAGIC   enmascara datos personales; `safety` bloquea contenido inseguro;
# MAGIC   `invalid_keywords` corta intentos comunes de prompt injection / jailbreak
# MAGIC   antes de que lleguen al modelo.
# MAGIC - **Rate limit**: protege la cuota de Free Edition — sin esto, un loop
# MAGIC   accidental en el cliente puede agotar el compute del día.
# MAGIC - **Usage tracking**: activa la inference table del endpoint (cada
# MAGIC   request/response queda logueado), base para monitoreo en producción.

# COMMAND ----------

guardrails = AiGatewayGuardrails(
    input=AiGatewayGuardrailParameters(
        safety=True,
        pii=AiGatewayGuardrailPiiBehavior(behavior=AiGatewayGuardrailPiiBehaviorBehavior.BLOCK),
        invalid_keywords=["ignore previous instructions", "ignore all previous instructions", "jailbreak"],
    ),
    output=AiGatewayGuardrailParameters(
        safety=True,
        pii=AiGatewayGuardrailPiiBehavior(behavior=AiGatewayGuardrailPiiBehaviorBehavior.BLOCK),
    ),
)

rate_limits = [
    AiGatewayRateLimit(
        calls=RATE_LIMIT_PER_MINUTE,
        key=AiGatewayRateLimitKey.ENDPOINT,
        renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
    )
]

usage_tracking_config = AiGatewayUsageTrackingConfig(enabled=True)

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
        config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=served_entities),
    )

try:
    w.serving_endpoints.put_ai_gateway(
        name=ENDPOINT_NAME,
        guardrails=guardrails,
        rate_limits=rate_limits,
        usage_tracking_config=usage_tracking_config,
    )
    print(f"Endpoint listo con guardrails + rate limit ({RATE_LIMIT_PER_MINUTE}/min) + usage tracking: {ENDPOINT_NAME}")
except Exception as e:  # noqa: BLE001
    # Usage tracking (inference tables) no está soportado para todos los tipos
    # de endpoint en todos los workspaces (p.ej. Free Edition). Que falte esa
    # feature no debería tumbar guardrails + rate limit, que sí son críticos.
    if "usage tracking" in str(e).lower():
        w.serving_endpoints.put_ai_gateway(
            name=ENDPOINT_NAME,
            guardrails=guardrails,
            rate_limits=rate_limits,
        )
        print(f"Endpoint listo con guardrails + rate limit ({RATE_LIMIT_PER_MINUTE}/min). Usage tracking no soportado en este workspace, se omitió: {ENDPOINT_NAME}")
    else:
        raise
