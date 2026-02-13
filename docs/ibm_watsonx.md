# IBM watsonx — The AI and Data Platform

IBM watsonx is IBM's integrated platform for AI development, data management, and AI governance. It is organized into three pillars, each addressing a distinct part of the AI lifecycle:

| Pillar | One-liner |
|---|---|
| **watsonx.ai** | Build, tune, and deploy foundation models |
| **watsonx.data** | Store and govern data for AI workloads |
| **watsonx.governance** | Monitor, explain, and govern AI in production |

This document covers each pillar in depth: what problems it solves, what functionalities it offers, and how to use it from Python.

**Table of Contents**

1. [watsonx.ai — Foundation Model Studio](#1-watsonxai--foundation-model-studio)
2. [watsonx.data — Data Lakehouse](#2-watsonxdata--data-lakehouse)
3. [watsonx.governance — AI Lifecycle Governance](#3-watsonxgovernance--ai-lifecycle-governance)
- [How the Three Pillars Connect](#how-the-three-pillars-connect)
- [General References](#general-references)

---

## 1. watsonx.ai — Foundation Model Studio

### What It Is

watsonx.ai is IBM's platform for working with foundation models (large language models and other generative AI models). It provides a web-based studio for interactive experimentation and a Python SDK for programmatic access.

### What Problems It Solves

- **Model access without infrastructure** — Run inference on large language models (LLMs) without provisioning GPUs or managing model serving. IBM hosts the models; you call an API.
- **Model selection** — Choose from IBM's own Granite model family, open-source models (Llama, Mistral, Mixtral), and models from the Hugging Face Hub. Experiment with different models for your use case without deploying each one yourself.
- **Prompt engineering** — Iterate on prompts interactively using the Prompt Lab UI, then export them to code.
- **Model customization** — Fine-tune models on your domain data using prompt tuning or other parameter-efficient methods, without needing deep ML expertise.

### Key Features

#### Prompt Lab

The Prompt Lab is the interactive web UI for prompt engineering. It offers three editing modes:

- **Structured mode** — Guided fields (instruction, examples, input) that help beginners construct effective prompts. The structured template teaches the model via example input-output pairs.
- **Freeform mode** — A blank text editor where you write the full prompt. For experienced prompt engineers who want full control.
- **Chat mode** — A conversational interface for testing dialog and question-answering behavior with multi-turn context.

The Prompt Lab also provides:
- A library of sample prompts for common tasks (classification, extraction, summarization, question answering)
- Session history to compare prompt variants
- Saved prompts that can be shared with team members
- Model parameter controls (temperature, top_p, top_k, max tokens, repetition penalty, decoding strategy)

#### Tuning Studio

Tuning Studio allows you to customize foundation models with your own data:

- **Prompt tuning** — Learns a small set of soft prompt tokens that steer the model's behavior, without modifying the model weights. Requires as few as 100-1,000 labeled examples. Fast and cost-effective — once tuned, the model can be used directly in Prompt Lab or deployed as an API endpoint.
- **Fine-tuning** — Full or parameter-efficient fine-tuning (LoRA) for deeper customization. Requires more data and compute but produces more significant behavior changes.
- **Training data formats** — Upload JSONL files with input-output pairs. Data can be sourced from Cloud Object Storage buckets.
- **Experiment tracking** — Compare training runs, view loss curves, and evaluate tuned models against baselines.

#### Model Catalog

The model catalog provides access to a broad range of models. The full, up-to-date list is maintained at [Supported foundation models](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx).

**IBM Granite models** — IBM's family of enterprise-grade, open-source foundation models. Granite is the first open model family to achieve ISO 42001 certification and scored 95% on Stanford's Foundation Model Transparency Index (highest ever recorded). The Granite 4.0 generation uses a hybrid Mamba-2/Transformer architecture that reduces memory requirements by 70%+ and doubles inference speed compared to pure Transformer models of the same size.

| Model | Parameters | Architecture | Key Capabilities |
|---|---|---|---|
| Granite 4.0 Nano | 350M, 1B | Dense | On-device deployment, minimal footprint |
| Granite 4.0 Micro | 3B | Dense Hybrid (Mamba-2/Transformer) | Function calling, local applications |
| Granite 4.0 Tiny | ~8B | Dense Hybrid | General purpose, efficient inference |
| Granite 4.0 Small (H) | 32B total / 9B active | MoE Hybrid | Instruction following, function calling, top benchmarks |
| Granite 3.2/3.3 | Various | Transformer | Vision language models (VLM) for document understanding |
| Granite Code | Various | Transformer | Fill-in-the-middle code completion, multi-language |
| Granite Time-Series | Various | Specialized | Time-series forecasting |
| Granite Embedding | e.g., 107M | Embedding | Multilingual text embeddings for RAG |

For agent development specifically, Granite 4.0 models are notable for their native function calling support — they achieved industry-leading results on the Berkeley Function Calling Leaderboard v3 (BFCLv3), meaning they reliably generate structured tool invocations when provided with tool schemas.

**Third-party open models** hosted on watsonx.ai infrastructure:
- **Meta Llama** — Llama 3.3 70B Instruct and other Llama variants
- **Mistral AI** — Mistral Large 3 (675B total / 41B active, 256k context), Mistral Large 2, Mixtral-8x7B

**Hugging Face integration** — Deploy select Hugging Face models directly into watsonx.ai.

#### Model Deployment and Inference

Models in watsonx.ai can be deployed as API endpoints for production use. Deployed models expose REST APIs for:
- **Text generation** (`/generation`) — Single-turn completions
- **Chat** (`/chat`) — Multi-turn conversational inference
- **Embeddings** (`/embeddings`) — Vector representations for RAG
- **Tokenization** (`/tokenization`) — Token counting and inspection

### Using watsonx.ai from Python

#### Authentication

All API calls require an IBM Cloud IAM API key and a watsonx.ai project ID. The API key is exchanged for a bearer token behind the scenes by the SDK.

```python
# Required environment variables:
# WATSONX_URL       — e.g., https://us-south.ml.cloud.ibm.com
# WATSONX_APIKEY    — your IBM Cloud IAM API key
# WATSONX_PROJECT_ID — your watsonx.ai project ID (found in the project settings)
```

You obtain these by:
1. Creating an IBM Cloud account and an IAM API key at [IBM Cloud API Keys](https://cloud.ibm.com/iam/apikeys)
2. Creating a watsonx.ai project at [dataplatform.cloud.ibm.com](https://dataplatform.cloud.ibm.com)
3. Copying the project ID from the project's *Manage* tab

#### Direct SDK Usage (`ibm_watsonx_ai`)

The `ibm_watsonx_ai` package is IBM's official Python SDK for watsonx.ai.

```python
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

# 1. Set up credentials
credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="your-api-key",
)

# 2. Initialize the model
model = ModelInference(
    model_id="ibm/granite-3-8b-instruct",   # or any model from the catalog
    credentials=credentials,
    project_id="your-project-id",
    params={
        "max_new_tokens": 500,
        "temperature": 0.7,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
    },
)

# 3. Generate text
response = model.generate_text("Explain what a RAG pipeline is in 3 sentences.")
print(response)
```

For streaming responses:

```python
for chunk in model.generate_text_stream("Explain RAG in detail."):
    print(chunk, end="")
```

For embeddings:

```python
from ibm_watsonx_ai.foundation_models import Embeddings

embedding_model = Embeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    credentials=credentials,
    project_id="your-project-id",
)

vectors = embedding_model.embed_documents(["First document", "Second document"])
```

#### LangChain Integration (`langchain-ibm`)

The `langchain-ibm` package provides LangChain-compatible wrappers around watsonx.ai, letting you plug IBM models into LangChain chains, agents, and RAG pipelines.

```python
from langchain_ibm import WatsonxLLM

llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="your-api-key",
    project_id="your-project-id",
    params={
        "max_new_tokens": 500,
        "temperature": 0.7,
    },
)

# Use like any LangChain LLM
response = llm.invoke("What is retrieval-augmented generation?")
```

For chat models (multi-turn):

```python
from langchain_ibm import ChatWatsonx
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="your-api-key",
    project_id="your-project-id",
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What are the three pillars of watsonx?"),
]
response = chat.invoke(messages)
```

For embeddings in LangChain:

```python
from langchain_ibm import WatsonxEmbeddings

embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="your-api-key",
    project_id="your-project-id",
)

vectors = embeddings.embed_documents(["text one", "text two"])
query_vector = embeddings.embed_query("search query")
```

### References — watsonx.ai

- [watsonx.ai — Product Page](https://www.ibm.com/products/watsonx-ai)
- [Supported Foundation Models (full catalog)](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-models.html?context=wx)
- [watsonx.ai Python SDK — API Reference (v1.5.1)](https://ibm.github.io/watsonx-ai-python-sdk/v1.5.1/api.html)
- [watsonx.ai Python SDK — GitHub](https://github.com/IBM/watsonx-ai-python-sdk)
- [watsonx.ai Sample Notebooks](https://github.com/IBM/watsonx-ai-samples)
- [Prompt Lab — Documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-prompt-lab.html?context=wx)
- [Prompt Lab — Tutorial](https://ibm.github.io/watsonx-prompt-lab/)
- [Tuning Studio — Documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-tuning-studio.html?context=wx)
- [Prompt Tuner SDK Reference](https://ibm.github.io/watsonx-ai-python-sdk/prompt_tuner.html)
- [IBM Granite Models](https://www.ibm.com/granite)
- [Granite 4.0 Documentation](https://www.ibm.com/granite/docs/models/granite)
- [Granite on Hugging Face](https://huggingface.co/ibm-granite)
- [langchain-ibm — PyPI](https://pypi.org/project/langchain-ibm/)
- [LangChain — ChatWatsonx Docs](https://python.langchain.com/docs/integrations/chat/ibm_watsonx/)
- [LangChain — WatsonxEmbeddings Docs](https://python.langchain.com/docs/integrations/text_embedding/ibm_watsonx/)
- [Foundation Model Supported Parameters](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-model-parameters.html?context=wx)

---

## 2. watsonx.data — Data Lakehouse

### What It Is

watsonx.data is IBM's open data lakehouse platform. A "lakehouse" combines the flexibility and cost-efficiency of a data lake (storing raw data in open file formats on cheap object storage) with the query performance and governance features of a data warehouse (schemas, ACID transactions, fine-grained access control).

### What Problems It Solves

- **Data silos** — Enterprises store data across many systems (relational databases, Hadoop clusters, cloud object storage, SaaS applications). watsonx.data provides a single query layer over all of them.
- **Data governance for AI** — AI models are only as good as their training data. watsonx.data provides metadata management, data lineage, and access controls so you know where your data came from, who can access it, and how it has been transformed.
- **Open formats and portability** — Data stored in Apache Iceberg, Parquet, Avro, and ORC formats avoids vendor lock-in. Any tool that reads these formats can access the data.

### Key Features

#### Query Engines

watsonx.data takes a "fit-for-purpose" approach — different query engines access the same data through a single entry point, and you pick the engine that best matches your workload:

- **Presto** — A distributed SQL query engine optimized for interactive analytics over large datasets. Presto can federate queries across multiple data sources (COS, Db2, PostgreSQL, Kafka, etc.) in a single SQL statement.
- **Apache Spark** — For batch processing, ETL, and complex transformations. Spark is better suited for large-scale data preparation (e.g., cleaning and transforming a document corpus before indexing into a vector database).

#### Vector Database Capabilities

watsonx.data integrates vector database services directly into the lakehouse:

- **Milvus** — Open-source vector database for similarity search, well-suited for RAG pipelines that need fast nearest-neighbor lookup over large embedding collections.
- **AstraDB** (Apache Cassandra-based) — Managed vector database for operational GenAI applications requiring high write/read throughput with global replication.

This means you can keep your document embeddings within the same governed data platform as your structured data — no need for a separate vector database deployment for production workloads.

#### Data Sources

watsonx.data can connect to:
- IBM Cloud Object Storage (COS)
- Amazon S3
- Relational databases (Db2, PostgreSQL, MySQL)
- Apache Kafka (Event Streams)
- Apache Hive metastore-compatible catalogs

#### Metadata and Catalog

watsonx.data uses an integrated metadata catalog that tracks:
- Table schemas and column types
- Data lineage (where data came from, how it was transformed)
- Access policies (who can query which tables)
- Statistics for query optimization

#### Integration with watsonx.ai

watsonx.data can feed data directly into watsonx.ai workflows:
- **Training data preparation** — Query, filter, and transform raw data into training datasets using SQL, then export to COS for model tuning.
- **RAG document sources** — Use SQL queries to select and filter documents from structured/semi-structured stores before feeding them into an embedding pipeline.
- **Feature computation** — Compute features or aggregations in watsonx.data and pass them as context to agent prompts.

### Using watsonx.data from Python

watsonx.data is primarily accessed via SQL through its Presto or Spark endpoints. You can connect using standard database drivers:

```python
# Using prestodb Python client
import prestodb

conn = prestodb.dbapi.connect(
    host="your-watsonx-data-hostname",
    port=443,
    user="your-username",
    catalog="your_catalog",
    schema="your_schema",
    http_scheme="https",
)

cursor = conn.cursor()
cursor.execute("SELECT * FROM documents WHERE category = 'technical' LIMIT 100")
rows = cursor.fetchall()
```

For Spark workloads, you interact via the Spark session configured in watsonx.data, or submit Spark applications through the watsonx.data console or API.

The `ibm_watsonx_ai` SDK also provides integration points for referencing watsonx.data assets in training jobs:

```python
# Reference a data asset from watsonx.data in a training configuration
training_data_references = [
    {
        "type": "connection_asset",
        "connection": {"id": "your-watsonx-data-connection-id"},
        "location": {"schema_name": "my_schema", "table_name": "training_data"},
    }
]
```

### References — watsonx.data

- [watsonx.data — Product Page](https://www.ibm.com/products/watsonx-data)
- [watsonx.data — Documentation](https://cloud.ibm.com/docs/watsonxdata?topic=watsonxdata-getting-started)
- [watsonx.data — GitHub Samples](https://github.com/IBM/watsonx-data)
- [Data Lakehouse Architecture Pattern](https://www.ibm.com/architectures/patterns/data-lakehouse)
- [Apache Iceberg — Project Site](https://iceberg.apache.org/)
- [Presto — Project Site](https://prestodb.io/)
- [Milvus — Project Site](https://milvus.io/)

---

## 3. watsonx.governance — AI Lifecycle Governance

### What It Is

watsonx.governance is IBM's platform for monitoring, explaining, and governing AI models and agents throughout their lifecycle. It provides dashboards, automated evaluations, and compliance tooling to ensure AI systems behave as expected in production.

### What Problems It Solves

- **Model drift** — Model performance degrades over time as the real-world data distribution shifts away from the training data. watsonx.governance monitors for data drift and accuracy drift, alerting you before users notice degradation.
- **Explainability** — Regulators, auditors, and end-users need to understand *why* a model made a particular decision. watsonx.governance generates human-readable explanations and feature importance scores for individual predictions.
- **Compliance and audit trails** — Industries like finance and healthcare require documentation of AI model development, testing, and deployment. watsonx.governance maintains an AI FactSheet — a structured record of each model's lifecycle.
- **Agent monitoring** — Agentic AI systems make chains of decisions. watsonx.governance can track agent decision paths, tool usage, and output quality over time — a capability that has become a major focus in 2025-2026.
- **Regulatory compliance** — Automates identification of regulatory requirements (EU AI Act, US Executive Order on AI, industry-specific regulations) and translates them into enforceable policies.

### Key Features

#### Fairness Monitoring

watsonx.governance monitors model predictions across protected attributes (gender, ethnicity, age, etc.) and computes fairness metrics:

- **Disparate Impact Ratio** — Measures whether favorable outcomes are distributed proportionally across groups
- **Statistical Parity Difference** — The difference in favorable outcome rates between groups
- **Direct and indirect bias detection** — Can detect bias even when the protected attribute is not a direct input to the model (indirect bias via correlated features)

When bias exceeds configured thresholds, watsonx.governance triggers alerts and can automatically generate de-biased predictions using its mitigation algorithms.

#### Quality Monitoring

Monitors model accuracy and performance metrics over time:

- **Classification metrics** — Accuracy, precision, recall, F1, area under ROC
- **Regression metrics** — MAE, RMSE, R-squared
- **Generative AI quality** — For LLM outputs: faithfulness, answer relevance, context relevance, harmfulness, PII detection
- **Custom metrics** — Define your own evaluation criteria

#### Drift Detection

- **Data drift** — Detects when the distribution of incoming data shifts significantly from the training data distribution (using statistical tests like KL divergence, Population Stability Index)
- **Model drift** — Detects when model predictions shift even if the input data appears stable (can indicate concept drift)
- **Configurable thresholds** — Set acceptable drift levels per feature and per metric

#### Explainability

Generates explanations for individual model predictions:

- **Feature importance** — Which input features most influenced the prediction (uses SHAP-like methods)
- **Contrastive explanations** — What would need to change in the input for the prediction to be different
- **Textual explanations** — Human-readable summaries of why a prediction was made

IBM also provides the open-source [AI Explainability 360 toolkit](https://aix360.res.ibm.com/) for deeper interpretability techniques.

#### AI FactSheets

AI FactSheets are structured documents that capture the complete lifecycle of an AI model:

- Model metadata (name, version, framework, training data description)
- Training details (hyperparameters, training metrics, dataset statistics)
- Evaluation results (accuracy, fairness, robustness)
- Deployment information (endpoint, version, deployment date)
- Ongoing monitoring results (drift, quality, fairness over time)

FactSheets provide the audit trail that regulators and compliance teams need.

#### Agent Governance

A major recent addition, agent governance extends watsonx.governance to monitor agentic AI applications specifically. Agents are treated as governed assets with continuous evaluation, policy enforcement, and automated block/route/fallback capabilities.

Agent-specific evaluation metrics:

| Metric | What It Measures |
|---|---|
| **Context Relevance** | How well retrieved data (RAG context) aligns with the user's prompt |
| **Faithfulness** | Whether the agent's response accurately reflects the retrieved context (hallucination detection) |
| **Answer Similarity** | How closely responses match predefined reference answers |
| **Tool Usage Patterns** | Which tools the agent calls, how often, and whether calls are appropriate |

Agent-specific security metrics detect threats unique to agentic systems:
- **Prompt injection** — Attempts to override the agent's system instructions via user input
- **Data exfiltration** — Detecting when an agent's tool calls might leak sensitive information
- **Unauthorized actions** — Flagging when an agent attempts actions outside its permitted scope

### Using watsonx.governance from Python

#### Setting up monitoring for a model

```python
from ibm_watsonx_ai import APIClient, Credentials

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="your-api-key",
)
client = APIClient(credentials)

# Subscribe a deployed model to governance monitoring
subscription = client.monitor_instances.create(
    data_mart_id="your-data-mart-id",
    service_provider_id="your-provider-id",
    asset={"asset_id": "your-model-deployment-id", "asset_type": "model"},
)
```

#### Configuring fairness monitoring

```python
# Configure fairness monitor
fairness_config = {
    "parameters": {
        "features": [
            {
                "feature": "age",
                "majority": [[26, 75]],
                "minority": [[18, 25]],
            }
        ],
        "favourable_class": ["approved"],
        "unfavourable_class": ["denied"],
        "min_records": 200,
    },
    "thresholds": [{"metric_id": "disparate_impact", "type": "lower_limit", "value": 0.8}],
}
```

#### Logging payload data for monitoring

```python
# Log scoring requests and responses for governance monitoring
client.monitor_instances.log_payload(
    subscription_id="your-subscription-id",
    request={"fields": ["age", "income", "credit_score"], "values": [[35, 50000, 720]]},
    response={"fields": ["prediction", "probability"], "values": [["approved", 0.87]]},
)
```

### References — watsonx.governance

- [watsonx.governance — Product Page](https://www.ibm.com/products/watsonx-governance)
- [watsonx.governance — Documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/model/getting-started.html?context=wx)
- [AI FactSheets — Documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/factsheets-model-inventory.html?context=wx)
- [Generative AI Quality Evaluations](https://dataplatform.cloud.ibm.com/docs/content/wsj/model/wos-monitor-gen-quality.html?context=wx)
- [Agentic AI Governance — Announcement](https://www.ibm.com/new/announcements/agentic-ai-governance-evaluation-and-lifecycle)
- [Governing AI Agents with watsonx.governance — Developer Guide](https://developer.ibm.com/articles/governing-ai-agents-watsonx-governance/)
- [Agent Security Metrics — Announcement](https://www.ibm.com/new/announcements/new-security-metrics-agent-monitoring-and-insights-in-watsonx-governance)
- [AI Explainability 360 — Open Source](https://aix360.res.ibm.com/)
- [AI Fairness 360 — Open Source](https://aif360.res.ibm.com/)

---

## How the Three Pillars Connect

The three watsonx pillars are designed to work together across the AI lifecycle:

```
    ┌─────────────────────────────────────────────────────────────┐
    │                     AI Lifecycle                            │
    │                                                             │
    │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
    │   │ watsonx.data │───▶│  watsonx.ai  │───▶│ watsonx.    │  │
    │   │              │    │              │    │ governance   │  │
    │   │  Prepare &   │    │  Build &     │    │  Monitor &   │  │
    │   │  govern data │    │  deploy      │    │  govern AI   │  │
    │   │              │    │  models      │    │              │  │
    │   └──────────────┘    └──────────────┘    └──────────────┘  │
    │         ▲                                       │           │
    │         └───────────── feedback ────────────────┘           │
    └─────────────────────────────────────────────────────────────┘
```

1. **watsonx.data** prepares and governs the data that feeds into model training and RAG pipelines
2. **watsonx.ai** uses that data to build, tune, and deploy foundation models
3. **watsonx.governance** monitors the deployed models and agents, detecting quality issues and bias
4. Governance findings feed back into data preparation and model retraining — closing the loop

---

## General References

- [IBM watsonx — Platform Overview](https://www.ibm.com/watsonx)
- [watsonx Developer Hub](https://www.ibm.com/watsonx/developer/)
- [ibm_watsonx_ai Python SDK — GitHub](https://github.com/IBM/watsonx-ai-python-sdk)
- [ibm_watsonx_ai Python SDK — API Docs (v1.5.1)](https://ibm.github.io/watsonx-ai-python-sdk/v1.5.1/api.html)
- [langchain-ibm — PyPI](https://pypi.org/project/langchain-ibm/)
- [watsonx on IBM Cloud — Getting Started](https://dataplatform.cloud.ibm.com/docs/content/wsj/getting-started/welcome-main.html?context=wx)
