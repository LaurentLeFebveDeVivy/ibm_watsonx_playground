# watsonx Agent Skeleton

A starter application kit that demonstrates how we could build a RAG-capable AI agent using IBM watsonx.ai, LangGraph, and FastAPI. Currentl, itt exposes two HTTP endpoints: one to ingest documents from IBM Cloud Object Storage into a Milvus vector database, and one to chat with a ReAct agent that can search the indexed knowledge base and query SQL databases.

## Running

From the repository root:

```bash
make run
```

This starts a FastAPI server on `http://localhost:8000` with hot reload enabled.

## Project Structure

```
watsonx-agent-skeletton/
├── main.py              # Entry point: creates the FastAPI app and runs uvicorn
├── requirements.txt     # Pinned Python dependencies
├── .python-version      # Enforces Python 3.12.3 via pyenv
├── config/
│   ├── config.py        # Credential loading and configuration
│   └── .env.example     # Template for required environment variables
└── app/
    ├── core/            # Shared infrastructure clients
    ├── agents/          # LangGraph agent definitions
    ├── prompts/         # Prompt templates
    ├── storage.py       # IBM Cloud Object Storage helpers
    ├── rag/             # RAG indexing and retrieval pipeline
    ├── tools/           # LangChain tool definitions for the agent
    ├── api/             # FastAPI app factory and route handlers
    ├── models/          # SQLAlchemy ORM models
    └── schemas/         # Pydantic request/response types
```

## `config/`

Handles all credential and environment configuration. `config.py` loads variables from a `config/.env` file (copied from `.env.example` during setup) and exposes them as frozen dataclasses — one per service:

| Dataclass | Environment variables | Purpose |
|---|---|---|
| `WatsonxCredentials` | `WATSONX_URL`, `WATSONX_APIKEY`, `WATSONX_PROJECT_ID` | IBM watsonx.ai API access |
| `COSCredentials` | `COS_ENDPOINT`, `COS_API_KEY`, `COS_INSTANCE_ID`, `COS_BUCKET_NAME` | IBM Cloud Object Storage |
| `MilvusCredentials` | `MILVUS_HOST`, `MILVUS_PORT`, `MILVUS_TOKEN`, `MILVUS_COLLECTION` | Milvus vector database |
| `DatabaseCredentials` | `DATABASE_URL` | SQL database (defaults to SQLite) |

## `app/`

### `core/` — Shared Infrastructure

Factory functions that create authenticated SDK clients. All other modules import from here rather than constructing clients directly.

- **`client.py`** — Singleton `APIClient` from `ibm_watsonx_ai`, cached with `@lru_cache` so the entire app shares one authenticated connection.
- **`llm.py`** — Creates `ChatWatsonx` (chat) and `WatsonxLLM` (text completion) instances. Default model: `ibm/granite-3-8b-instruct`.
- **`embeddings.py`** — Creates `WatsonxEmbeddings` instances. Default model: `ibm/slate-125m-english-rtrvr-v2`.

### `rag/` — Retrieval-Augmented Generation Pipeline

The indexing and retrieval logic that lets the agent answer questions from uploaded documents.

- **`vectorstore.py`** — Builds a `langchain_milvus.Milvus` vectorstore connected to a Milvus instance.
- **`indexing.py`** — Downloads documents from IBM COS, splits them into chunks (1000 chars, 200 overlap), embeds them, and stores the vectors in Milvus.
- **`retrieval.py`** — Searches the vectorstore for relevant chunks. Uses a two-stage approach: fetches 20 candidates via vector similarity, then reranks them with `WatsonxRerank` (cross-encoder) and returns the top 4.

### `tools/` — Agent Tools

LangChain tool definitions that the agent can invoke during reasoning.

- **`search.py`** — `search_knowledge_base` tool: wraps the RAG retrieval pipeline so the agent can look up information from indexed documents.
- **`database.py`** — Builds SQL tools via `WatsonxSQLDatabaseToolkit` (list tables, get schema, run queries, check queries). Uses lazy imports to avoid loading heavy dependencies when database tools are not needed.

### `agents/` — Agent Definitions

- **`rag_agent.py`** — Builds a ReAct agent using LangGraph's `create_react_agent()`. The agent always has the knowledge base search tool and can optionally receive additional database tools.

### `prompts/` — Prompt Templates

- **`templates.py`** — System prompt that instructs the agent to use its tools and not fabricate answers, a `ChatPromptTemplate` for the agent, and a helper function that formats retrieved documents into numbered excerpts with source attribution.

### `api/` — FastAPI Web Layer

- **`app.py`** — App factory (`create_app()`) with a lifespan handler that initializes the database on startup.
- **`routes/chat.py`** — `POST /chat`: accepts a message (and optional session ID), invokes the agent, and returns the reply.
- **`routes/ingest.py`** — `POST /ingest`: accepts a COS prefix and collection name, runs the indexing pipeline, and returns the number of documents indexed.

### `models/` — Database Models

- **`base.py`** — SQLAlchemy `DeclarativeBase`, engine creation (configured from `DATABASE_URL`), session factory, and `init_db()` to create tables on startup.

### `schemas/` — Request/Response Types

Pydantic models for API validation:

- **`chat.py`** — `ChatRequest` (message, optional session_id) and `ChatResponse` (reply, session_id).
- **`ingest.py`** — `IngestRequest` (cos_prefix, collection_name) and `IngestResponse` (documents_indexed, collection_name).
