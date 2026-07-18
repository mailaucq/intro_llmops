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

import mlflow
from mlflow.models.resources import DatabricksServingEndpoint
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

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

class PythonQAAgent(ResponsesAgent):
    """Envuelve prompt@production + el endpoint LLM detrás de la interfaz estándar."""

    def __init__(self, model_endpoint: str, prompt_name: str, prompt_alias: str = "production"):
        self.model_endpoint = model_endpoint
        self.prompt_name = prompt_name
        self.prompt_alias = prompt_alias

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        import mlflow.genai
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        llm = w.serving_endpoints.get_open_ai_client()
        prompt = mlflow.genai.load_prompt(f"prompts:/{self.prompt_name}@{self.prompt_alias}")

        question = request.input[-1].content
        if isinstance(question, list):
            question = "".join(
                part.get("text", "") for part in question if isinstance(part, dict)
            )

        messages = prompt.format(question=question)
        response = llm.chat.completions.create(
            model=self.model_endpoint,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        output_text = response.choices[0].message.content

        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=output_text, id=str(uuid.uuid4()))]
        )

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
