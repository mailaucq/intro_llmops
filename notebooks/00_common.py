# Databricks notebook source
# MAGIC %md
# MAGIC ## Código compartido del pipeline
# MAGIC
# MAGIC Se incluye con `%run ./00_common` desde `01_eval_baseline` y `02_eval_candidate`.
# MAGIC No instala paquetes ni reinicia Python — eso lo hace cada notebook que lo llama,
# MAGIC antes del `%run`, porque `dbutils.library.restartPython()` borra el estado.

# COMMAND ----------

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SOURCE_JSONL = dbutils.widgets.get("source_jsonl")
MODEL_ENDPOINT = dbutils.widgets.get("model_endpoint")

import threading
import time

import mlflow
import mlflow.genai
from databricks.sdk import WorkspaceClient
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery, Safety, scorer

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

PROMPT_NAME = f"{CATALOG}.{SCHEMA}.python_qa"
JUDGE_MODEL = "databricks-qwen35-122b-a10b"

w = WorkspaceClient()
llm = w.serving_endpoints.get_open_ai_client()

mlflow.set_experiment(f"/Users/{w.current_user.me().user_name}/python_qa_llmops")
mlflow.openai.autolog()

# COMMAND ----------

# MAGIC %md ### Datos de evaluación — el `.jsonl` del lab, o 4 ejemplos de arranque

# COMMAND ----------

N_EVAL = 20

if SOURCE_JSONL:
    rows = spark.read.json(SOURCE_JSONL).limit(N_EVAL).collect()
    eval_data = [
        {
            "inputs": {"question": r["input_text"]},
            "expectations": {"expected_response": r["output_text"]},
        }
        for r in rows
    ]
else:
    eval_data = [
        {
            "inputs": {"question": "How do I reverse a list in Python?"},
            "expectations": {"expected_response":
                "Use list.reverse() to reverse in place, or the slice [::-1] "
                "to get a reversed copy. reversed() returns an iterator."},
        },
        {
            "inputs": {"question": "What is the difference between a list and a tuple?"},
            "expectations": {"expected_response":
                "Lists are mutable and use square brackets; tuples are immutable "
                "and use parentheses. Tuples can be dict keys, lists cannot."},
        },
        {
            "inputs": {"question": "How do I read a JSON file in Python?"},
            "expectations": {"expected_response":
                "Open the file and use json.load(f). For a JSON string, use json.loads()."},
        },
        {
            "inputs": {"question": "Why does mutable default argument cause bugs?"},
            "expectations": {"expected_response":
                "Default arguments are evaluated once at function definition, so a "
                "mutable default is shared across calls. Use None as the default and "
                "create the object inside the function."},
        },
    ]

print(f"{len(eval_data)} ejemplos de evaluación")

# COMMAND ----------

# MAGIC %md ### La app — carga el prompt por alias `champion`, no hardcodeado

# COMMAND ----------

semaphore = threading.Semaphore(2)

@mlflow.trace(span_type="CHAIN")
def python_qa(question: str) -> str:
    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@champion")
    messages = prompt.format(question=question)
    with semaphore:
        response = llm.chat.completions.create(
            model=MODEL_ENDPOINT,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        time.sleep(0.2)
    return response.choices[0].message.content

# COMMAND ----------

# MAGIC %md ### Scorers — código primero, judges solo donde la regla no se puede escribir

# COMMAND ----------

@scorer
def has_code_block(outputs: str) -> bool:
    """Una respuesta de programación debería mostrar código."""
    return "```" in outputs

@scorer
def is_concise(outputs: str) -> int:
    """Métrica numérica: palabras. Para ver la deriva de verbosidad entre versiones."""
    return len(outputs.split())

scorers = [
    Correctness(model=JUDGE_MODEL),
    RelevanceToQuery(model=JUDGE_MODEL),
    Safety(model=JUDGE_MODEL),
    Guidelines(
        name="answers_directly",
        guidelines=(
            "The response must answer the question directly without preamble "
            "such as 'Great question!' or restating the question."
        ),
        model=JUDGE_MODEL,
    ),
    has_code_block,
    is_concise,
]

# COMMAND ----------

def tidy_metrics(metrics: dict) -> dict:
    """{'correctness/mean': 0.8, ...} -> {'correctness': 0.8, ...}, solo numéricos."""
    return {
        k.replace("/mean", ""): float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float))
    }
