# Databricks notebook source
# MAGIC %md
# MAGIC # 4. Registrar el modelo — ResponsesAgent en Unity Catalog
# MAGIC
# MAGIC Solo corre si `gate_check` dice `true`. Empaqueta el prompt
# MAGIC `@production` + el endpoint LLM como un `mlflow.pyfunc.ResponsesAgent`
# MAGIC (el estándar de Databricks para agentes servibles — schema compatible
# MAGIC con la Responses API, streaming, tracing y `mlflow.genai.evaluate`).
# MAGIC
# MAGIC Se declara `resources=[DatabricksServingEndpoint(...)]` para que, al
# MAGIC servir este modelo, la autenticación hacia el endpoint LLM subyacente
# MAGIC se resuelva automáticamente (sin secretos hardcodeados).

# COMMAND ----------

# MAGIC %pip install -q --upgrade "mlflow[databricks]>=3.1.0" openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "llmops")
dbutils.widgets.text("model_endpoint", "databricks-qwen35-122b-a10b")

# COMMAND ----------

import uuid
from typing import Generator

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_ENDPOINT = dbutils.widgets.get("model_endpoint")

PROMPT_NAME = f"{CATALOG}.{SCHEMA}.python_qa"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.qa_model"

winner_version = dbutils.jobs.taskValues.get(
    taskKey="compare_and_gate", key="winner_version", default=None, debugValue=1
)

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

def _extract_text(content) -> str:
    """Modelos 'reasoning' devuelven content como lista de bloques
    ([{'type': 'reasoning', ...}, {'type': 'text', ...}]) en vez de un string.
    Nos quedamos solo con los bloques de texto, en orden."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") in ("text", "output_text")
        ]
        return "".join(parts)
    return str(content)

class PythonQAAgent(ResponsesAgent):
    """Envuelve prompt@production + el endpoint LLM detrás de la interfaz estándar.

    Cache exact-match en memoria: mismo texto de pregunta -> misma respuesta,
    sin volver a llamar al LLM. Vive por réplica del serving endpoint, se
    pierde en cada redeploy/restart/scale-to-zero; para que persista entre
    réplicas habría que respaldarlo en una tabla Delta en vez de un dict.
    """

    def __init__(self, model_endpoint: str, prompt_name: str, prompt_alias: str = "production"):
        self.model_endpoint = model_endpoint
        self.prompt_name = prompt_name
        self.prompt_alias = prompt_alias
        self._cache: dict[str, str] = {}

    def _cache_key(self, question: str) -> str:
        import hashlib
        return hashlib.sha256(f"{self.prompt_alias}:{question}".encode()).hexdigest()

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        import mlflow
        import mlflow.genai
        from databricks.sdk import WorkspaceClient

        mlflow.set_registry_uri("databricks-uc")
        mlflow.openai.autolog()
        mlflow.set_tracking_uri("databricks")
        question = request.input[-1].content
        if isinstance(question, list):
            question = "".join(
                part.get("text", "") for part in question if isinstance(part, dict)
            )

        cache_key = self._cache_key(question)
        output_text = self._cache.get(cache_key)

        if output_text is None:
            w = WorkspaceClient()
            llm = w.serving_endpoints.get_open_ai_client()
            prompt = mlflow.genai.load_prompt(f"prompts:/{self.prompt_name}@{self.prompt_alias}")
            messages = prompt.format(question=question)
            response = llm.chat.completions.create(
                model=self.model_endpoint,
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            output_text = _extract_text(response.choices[0].message.content)
            self._cache[cache_key] = output_text

        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=output_text, id=str(uuid.uuid4()))]
        )

    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Sin token-streaming real desde el LLM subyacente (una sola llamada
        bloqueante a chat.completions.create, reusada por el cache). Se emite
        la respuesta completa como un único evento para que el endpoint no
        falle cuando lo consultan con stream=True (p.ej. el tester de
        Databricks Model Serving)."""
        response = self.predict(request)
        for item in response.output:
            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)

# COMMAND ----------

agent = PythonQAAgent(model_endpoint=MODEL_ENDPOINT, prompt_name=PROMPT_NAME, prompt_alias="production")

input_example = {"input": [{"role": "user", "content": "How do I reverse a list in Python?"}]}

with mlflow.start_run(run_name="register_qa_model"):
    logged_model = mlflow.pyfunc.log_model(
        name="agent",
        python_model=agent,
        input_example=input_example,
        registered_model_name=UC_MODEL_NAME,
        resources=[DatabricksServingEndpoint(endpoint_name=MODEL_ENDPOINT)],
        pip_requirements=["mlflow[databricks]>=3.1.0", "openai"],
    )

registered_version = logged_model.registered_model_version
print(f"{UC_MODEL_NAME} v{registered_version} <- prompt {PROMPT_NAME}@production (v{winner_version})")
print("model_uri:", logged_model.model_uri)

# COMMAND ----------

dbutils.jobs.taskValues.set("registered_model_version", registered_version)
