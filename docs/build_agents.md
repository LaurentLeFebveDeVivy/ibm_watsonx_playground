# Building AI Agents — A Practical Guide

This guide explains what an AI agent is, what components it needs, and how to build one using IBM products and complementary open-source software.

**Table of Contents**

1. [What Is an AI Agent?](#1-what-is-an-ai-agent)
2. [Components of an AI Agent](#2-components-of-an-ai-agent)
   - 2.1 [Foundation Model (LLM)](#21-foundation-model-llm)
   - 2.2 [Tools](#22-tools)
   - 2.3 [RAG (Retrieval-Augmented Generation)](#23-rag-retrieval-augmented-generation)
   - 2.4 [Memory](#24-memory)
   - 2.5 [Orchestration Framework](#25-orchestration-framework)
   - 2.6 [Prompt Templates](#26-prompt-templates)
   - 2.7 [Guardrails and Safety](#27-guardrails-and-safety)
3. [Agent Architectures](#3-agent-architectures)
   - 3.1 [Simple ReAct Agent](#31-simple-react-agent)
   - 3.2 [RAG Agent](#32-rag-agent)
   - 3.3 [Multi-Agent Systems](#33-multi-agent-systems)
   - 3.4 [Human-in-the-Loop Agent](#34-human-in-the-loop-agent)
4. [Building an Agent — Technology Stack](#4-building-an-agent--technology-stack)
5. [Putting It Together — Agent with LangGraph](#5-putting-it-together--agent-with-langgraph)
6. [watsonx Orchestrate — Enterprise Agent Management](#6-watsonx-orchestrate--enterprise-agent-management)
7. [Development Workflow](#7-development-workflow)
- [References](#references)

---

## 1. What Is an AI Agent?

An AI agent is a software system that uses a large language model (LLM) as its reasoning engine to autonomously plan and execute multi-step tasks. Unlike a simple chatbot that produces a single response to a single prompt, an agent:

- **Reasons** about what steps are needed to accomplish a goal
- **Plans** a sequence of actions
- **Executes** those actions by calling tools (APIs, databases, code execution, etc.)
- **Observes** the results of its actions
- **Iterates** — adjusts its plan based on what it learned, and takes further actions until the task is complete

The core loop of an agent looks like this:

```
User query
    │
    ▼
┌─────────────────────────────────┐
│         Agent Loop              │
│                                 │
│  1. THINK  — LLM reasons about  │
│             the current state   │
│             and decides what    │
│             to do next          │
│                                 │
│  2. ACT   — Agent calls a tool  │
│             (search, API call,  │
│             code execution,     │
│             database query)     │
│                                 │
│  3. OBSERVE — Agent receives    │
│              the tool's output  │
│              and feeds it back  │
│              to the LLM         │
│                                 │
│  Repeat until task is complete  │
└─────────────────────────────────┘
    │
    ▼
Final response to user
```

This is often called the **ReAct pattern** (Reasoning + Acting), introduced in the [ReAct paper (Yao et al., 2022)](https://arxiv.org/abs/2210.03629).

### Agent vs. Chain vs. Chatbot

| System | Behavior | Example |
|---|---|---|
| **Chatbot** | Single-turn: receives a message, produces a response. No tool use, no memory across turns. | "What's the weather?" → "I don't have access to weather data." |
| **Chain** | Fixed sequence of steps: prompt → LLM → parser → output. Steps are predetermined at build time. | Summarize a document → extract entities → format as JSON |
| **Agent** | Dynamic: the LLM decides which tools to call and in what order, based on the task. The execution path is not predetermined. | "Find all invoices from last month, calculate the total, and email me a summary." → Agent searches database, does arithmetic, sends email. |

---

## 2. Components of an AI Agent

Every agent system is composed of the following building blocks. Not all are required for every agent.

### 2.1 Foundation Model (LLM)

The brain of the agent. The LLM receives the current conversation context (including tool results) and decides what to do next. The quality of the model directly determines the agent's ability to reason, follow instructions, and use tools correctly.

**Key requirements for an agent LLM:**
- **Instruction following** — The model must reliably follow system prompts and tool-calling conventions
- **Tool/function calling** — The model must be able to generate structured tool invocations (usually JSON) rather than free-text responses when appropriate
- **Context window** — Larger context windows allow the agent to hold more conversation history, tool results, and retrieved documents in a single call
- **Reasoning ability** — The model must be able to plan multi-step solutions and correct course when a step fails

**Technologies:**

| Technology | Type | Notes |
|---|---|---|
| **IBM Granite 4.0 models** (via watsonx.ai) | IBM, open-source (Apache 2.0) | Enterprise-grade, ISO 42001 certified. Industry-leading function calling on BFCLv3 benchmark. Hybrid Mamba-2/Transformer architecture for 70%+ lower memory, 2x faster inference. Sizes from 350M (Nano) to 32B (Small H, MoE). |
| **Llama, Mistral, Mixtral** (via watsonx.ai) | Open-source, IBM-hosted | Open models hosted on watsonx.ai infrastructure, accessible via the same SDK. Mistral Large 3 offers 256k context window. |
| **Self-hosted models** | Open-source | Run any Hugging Face model on your own infrastructure using `transformers`, `vLLM`, or `TGI`. Requires GPU hardware. |

See [ibm_watsonx.md](ibm_watsonx.md) for details on using watsonx.ai models from Python.

### 2.2 Tools

Tools are functions that the agent can invoke to interact with the outside world. They are what separate an agent from a pure text generator. Tools can be anything with a programmatic interface:

- **Search / retrieval** — Query a vector database, search the web, look up a knowledge base
- **Database operations** — Run SQL queries, insert records, read from key-value stores
- **API calls** — Call REST APIs (weather, CRM, payment processing, etc.)
- **Code execution** — Run Python code in a sandbox, execute shell commands
- **File operations** — Read, write, or parse files (PDFs, CSVs, spreadsheets)
- **Communication** — Send emails, post Slack messages, create calendar events

Each tool is defined with:
1. A **name** — Unique identifier (e.g., `search_documents`)
2. A **description** — Natural language explanation of what the tool does and when to use it. This is what the LLM reads to decide which tool to call.
3. A **schema** — The parameters the tool accepts (names, types, descriptions). Usually defined as a JSON Schema or Pydantic model.
4. An **implementation** — The actual Python function that executes when called.

**Technologies:**

| Technology | Role |
|---|---|
| **LangChain Tools** | Framework for defining tools with `@tool` decorator or `StructuredTool` class. Provides pre-built tools for common operations (web search, SQL, file I/O, etc.) |
| **IBM App Connect** | No-code connectors to enterprise systems (Salesforce, SAP, Slack). Can be exposed as REST endpoints that the agent calls as tools. |
| **Custom Python functions** | Any Python function can be wrapped as a tool. This is the most flexible approach. |

**Example — defining a tool in LangChain:**

```python
from langchain_core.tools import tool

@tool
def search_documents(query: str) -> str:
    """Search the document knowledge base for information relevant to the query.
    Use this tool when you need to find specific facts, policies, or procedures."""
    # Implementation: query a vector database
    results = vector_store.similarity_search(query, k=5)
    return "\n\n".join([doc.page_content for doc in results])

@tool
def get_customer_info(customer_id: str) -> str:
    """Look up customer information by ID. Returns name, email, and account status."""
    # Implementation: query a database
    row = db.execute("SELECT * FROM customers WHERE id = ?", [customer_id]).fetchone()
    return str(dict(row)) if row else "Customer not found."
```

### 2.3 RAG (Retrieval-Augmented Generation)

RAG is a pattern that gives the agent access to a knowledge base of documents. Instead of relying solely on the LLM's training data (which is static and may be outdated), RAG retrieves relevant documents at runtime and includes them in the LLM's context.

The RAG pipeline has three stages:

#### Indexing (offline, done once or periodically)

1. **Load** documents from sources (files, databases, web pages, APIs)
2. **Split** documents into smaller chunks (e.g., 500-1000 tokens each) so they fit in the LLM's context window
3. **Embed** each chunk — convert it to a dense vector using an embedding model
4. **Store** the vectors in a vector database alongside the original text

#### Retrieval (at query time)

1. **Embed** the user's query using the same embedding model
2. **Search** the vector database for chunks whose vectors are most similar to the query vector (cosine similarity, dot product, etc.)
3. **Return** the top-k most relevant chunks

#### Augmentation and Generation

1. **Augment** the prompt by inserting the retrieved chunks as context
2. **Generate** the answer using the LLM, which now has access to relevant, up-to-date information

**Technologies:**

| Component | Technology Options |
|---|---|
| **Document loading** | LangChain document loaders (PDF, HTML, Markdown, CSV, JSON, databases) |
| **Text splitting** | LangChain text splitters (`RecursiveCharacterTextSplitter`, `TokenTextSplitter`) |
| **Embedding model** | IBM Granite Embedding (via watsonx.ai), `sentence-transformers` (local, open-source), OpenAI embeddings |
| **Vector database** | ChromaDB (lightweight, embedded), Milvus (scalable, distributed), Elasticsearch (also does keyword search), Watson Discovery (IBM managed) |
| **LLM for generation** | IBM Granite models or other models via watsonx.ai |

**Example — building a RAG pipeline:**

```python
from langchain_ibm import WatsonxEmbeddings, ChatWatsonx
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load documents
loader = PyPDFLoader("company_handbook.pdf")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# 3. Create embeddings and store in vector DB
embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="your-api-key",
    project_id="your-project-id",
)
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. Retrieve relevant context at query time
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
relevant_docs = retriever.invoke("What is the vacation policy?")

# 5. Generate answer with context
llm = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="your-api-key",
    project_id="your-project-id",
)

context = "\n\n".join([doc.page_content for doc in relevant_docs])
prompt = f"""Answer the question based on the following context.

Context:
{context}

Question: What is the vacation policy?

Answer:"""

response = llm.invoke(prompt)
```

### 2.4 Memory

Memory allows the agent to retain information across interactions. Without memory, every conversation turn starts from scratch.

Types of memory:

- **Conversation memory** — The history of messages in the current session. The simplest form: just append all messages to the context window.
- **Summary memory** — A compressed summary of past conversations, useful when the full history would exceed the context window.
- **Persistent memory** — Facts stored in a database that survive across sessions. For example, remembering a user's preferences or the outcome of a previous task.
- **Checkpointing** — Saving the full state of an agent graph at each step, enabling pause/resume and replay of agent execution.

**Technologies:**

| Technology | Memory Type |
|---|---|
| **LangGraph Checkpointing** | Full state persistence — saves the agent graph state (messages, tool results, intermediate decisions) at each node. Enables pause/resume, time-travel debugging, and human-in-the-loop workflows. |
| **LangChain Memory classes** | Conversation buffer, conversation summary, entity memory. In-memory or backed by databases. |
| **SQLite / PostgreSQL** | Persistent storage for checkpoints and conversation history. LangGraph provides `SqliteSaver` and `PostgresSaver` out of the box. |
| **Redis** | Fast key-value store for session state and caching. |

### 2.5 Orchestration Framework

The orchestration framework ties all the components together: it manages the agent loop, routes between LLM calls and tool executions, handles errors, and enforces control flow.

**Technologies:**

| Technology | Description |
|---|---|
| **LangGraph** | A graph-based orchestration framework built on LangChain. Agents are defined as directed graphs where nodes are functions (LLM calls, tool executions, logic) and edges define the flow between them. Supports cycles (the agent loop), conditional branching, parallel execution, human-in-the-loop, and checkpointing. This is the recommended framework for production agents. |
| **LangChain AgentExecutor** | The simpler, older approach. A single loop that alternates between LLM and tool calls. Less flexible than LangGraph but easier to set up for simple cases. |
| **IBM Agent Lab** | A low-code visual builder in the watsonx.ai console. Build agents by configuring components through a web UI. Good for prototyping. |
| **IBM watsonx Orchestrate** | See below for description: [watsonx Orchestrate](#6-watsonx-orchestrate--enterprise-agent-management)|
### 2.6 Prompt Templates

Prompt templates define how the agent's system instructions, user input, tool descriptions, and retrieved context are assembled into the actual text sent to the LLM. Good prompt templates are critical — they determine whether the agent uses tools correctly, follows instructions, and produces useful outputs.

Key prompts in an agent system:
- **System prompt** — Defines the agent's role, capabilities, constraints, and behavior guidelines
- **Tool description prompt** — Explains each available tool to the LLM (auto-generated from tool definitions in LangChain)
- **RAG context prompt** — Template for inserting retrieved documents into the generation prompt
- **Few-shot examples** — Example interactions that demonstrate the expected behavior

### 2.7 Guardrails and Safety

Guardrails prevent the agent from taking harmful, unintended, or out-of-scope actions.

Types of guardrails:
- **Input validation** — Filter or reject user inputs that are out of scope, harmful, or attempting prompt injection
- **Output validation** — Check LLM outputs before returning them to the user (e.g., filter PII, ensure factual grounding)
- **Tool restrictions** — Limit which tools the agent can call, and validate tool arguments before execution
- **Rate limiting** — Prevent runaway agent loops (set max iterations, max tokens, timeouts)
- **Human-in-the-loop** — Require human approval before the agent executes high-impact actions (e.g., sending emails, modifying data)

**Technologies:**

| Technology | Role |
|---|---|
| **LangGraph interrupt/approval nodes** | Pause the agent graph and wait for human approval before proceeding. Built into the graph definition. |
| **watsonx.governance** | Production monitoring for bias, drift, and quality. See [ibm_watsonx.md](ibm_watsonx.md). |
| **Custom validation functions** | Python functions that validate inputs/outputs at each step. |

---

## 3. Agent Architectures

### 3.1 Simple ReAct Agent

The most basic architecture. A single LLM in a loop with tools.

```
User → [LLM → Tool → LLM → Tool → ... → LLM] → Response
```

Best for: Single-purpose agents with a small number of tools.

### 3.2 RAG Agent

A ReAct agent where one of the primary tools is a retrieval function that queries a knowledge base. The agent decides when to search for information and how to incorporate it.

```
User → [LLM → Retrieve → LLM → (maybe Retrieve again) → LLM] → Response
```

Best for: Question-answering over a document corpus, support agents, research assistants.

### 3.3 Multi-Agent Systems

Multiple specialized agents collaborate on a task. Each agent has its own tools and expertise. A supervisor agent (or a predefined routing mechanism) delegates sub-tasks.

```
User → [Supervisor Agent]
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
  Research  Code   Review
   Agent   Agent   Agent
```

Best for: Complex tasks that require different types of expertise (research, coding, analysis, communication).

### 3.4 Human-in-the-Loop Agent

The agent pauses at designated checkpoints and waits for human approval before continuing. LangGraph supports this natively with interrupt nodes.

Best for: High-stakes operations (financial transactions, data modifications, external communications).

---

## 4. Building an Agent — Technology Stack

Here is how the abstract agent components map to concrete technologies available in the IBM ecosystem and open-source world.

### Recommended Stack

| Component | Technology | Why |
|---|---|---|
| **LLM** | IBM Granite (via watsonx.ai) | Enterprise-grade, function calling support, IBM-hosted inference, no GPU management |
| **LLM Integration** | `langchain-ibm` (`ChatWatsonx`) | Plugs watsonx.ai models into LangChain/LangGraph seamlessly |
| **Orchestration** | LangGraph | Graph-based agent loops with cycles, branching, checkpointing, human-in-the-loop |
| **Embedding Model** | `sentence-transformers` (local) or Granite Embedding (watsonx.ai) | Local option avoids API calls during indexing; IBM option for consistent managed infrastructure |
| **Vector Database** | ChromaDB | Lightweight, embedded, Python-native. Good for development and moderate-scale production. Scales up with Milvus or Elasticsearch. |
| **Document Processing** | LangChain document loaders + text splitters | Handles PDF, HTML, Markdown, CSV, and more. Splits documents into retrieval-friendly chunks. |
| **API Framework** | FastAPI | High-performance async Python web framework. Exposes the agent as a REST API. |
| **Memory / State** | LangGraph checkpointing (SQLite or PostgreSQL) | Persists agent state across turns. Enables pause/resume and debugging. |
| **Monitoring** | watsonx.governance | Production monitoring for bias, drift, quality. |
| **Deployment** | IBM Code Engine or IKS | Serverless (Code Engine) for simplicity, Kubernetes (IKS) for control. |

### Python Dependencies

These are the core packages for building a watsonx-based agent:

```
# IBM SDKs
ibm_watsonx_ai         # Direct watsonx.ai SDK
langchain-ibm           # LangChain wrappers for watsonx.ai

# LangChain / LangGraph
langchain               # Core LangChain framework
langchain-core          # Base abstractions
langchain-community     # Community integrations (vector stores, doc loaders)
langgraph               # Agent orchestration framework

# RAG components
chromadb                # Vector database
sentence-transformers   # Local embedding models

# API server
fastapi                 # Web framework
uvicorn                 # ASGI server

# Supporting
python-dotenv           # Environment variable management
pydantic                # Data validation and settings
```

---

## 5. Putting It Together — Agent with LangGraph

Below is a conceptual example of building a ReAct agent with RAG using LangGraph, watsonx.ai, and ChromaDB. This illustrates how all the components connect.

### Step 1: Define the LLM

```python
from langchain_ibm import ChatWatsonx

llm = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="your-api-key",
    project_id="your-project-id",
    params={"max_new_tokens": 1024, "temperature": 0.0},
)
```

### Step 2: Define tools

```python
from langchain_core.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for relevant information.
    Use this when you need to look up facts, policies, or procedures."""
    results = vector_store.similarity_search(query, k=5)
    return "\n\n".join([doc.page_content for doc in results])

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

tools = [search_knowledge_base, calculate]
```

Note: You can also bind tools directly to the chat model for explicit tool calling:

```python
llm_with_tools = llm.bind_tools(tools)
```

This tells the LLM about the available tools and their schemas, so it can generate structured tool invocations during the agent loop.

### Step 3: Build the agent graph with LangGraph

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Create a checkpointer for conversation persistence
checkpointer = MemorySaver()

# Build the agent — LangGraph handles the ReAct loop internally
agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=checkpointer,
)
```

### Step 4: Run the agent

```python
# Invoke the agent with a user query
config = {"configurable": {"thread_id": "session-001"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "What is our refund policy for items over $100?"}]},
    config=config,
)

# The agent will:
# 1. Decide to search the knowledge base
# 2. Read the retrieved documents
# 3. Formulate an answer based on the retrieved context
print(response["messages"][-1].content)
```

### Step 5: Expose as an API

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    message: str
    session_id: str = "default"

@app.post("/chat")
async def chat(query: Query):
    config = {"configurable": {"thread_id": query.session_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": query.message}]},
        config=config,
    )
    return {"response": response["messages"][-1].content}
```

---


## 6. watsonx Orchestrate — Enterprise Agent Management

For organizations managing multiple agents at scale, **watsonx Orchestrate** provides the Agent Development Kit (ADK) for lifecycle management, deployment, and orchestration of agents. Key capabilities:

- Deploy agents built in watsonx.ai or from external frameworks (LangGraph, Langflow, custom)
- Register deployed agents as "collaborators" that can call each other
- Manage agent versions, permissions, and routing
- Build and deploy agents directly from your IDE

watsonx Orchestrate is aimed at enterprise scenarios where multiple agents cooperate across business functions (HR, IT, finance) and need centralized management.

- [watsonx Orchestrate — AI Agent Builder](https://www.ibm.com/products/watsonx-orchestrate/ai-agent-builder)
- [Build Custom AI Agents with Langflow and watsonx Orchestrate](https://www.ibm.com/think/tutorials/build-custom-ai-agents-with-langflow)
- [Build and Deploy Agents from Your IDE](https://www.ibm.com/new/announcements/build-and-deploy-agents-to-watsonx-ai-from-your-ide)

---

## 7. Development Workflow

A typical workflow for building an agent:

1. **Define the use case** — What task should the agent accomplish? What tools does it need?
2. **Set up credentials** — Create an IBM Cloud account, generate an IAM API key, create a watsonx.ai project. Store credentials in a `.env` file.
3. **Prototype in Prompt Lab** — Use the watsonx.ai Prompt Lab to experiment with different models and prompts. Find a model + prompt combination that performs well for your task.
4. **Build the RAG pipeline** (if needed) — Load documents, create embeddings, store in a vector database. Test retrieval quality independently before connecting to the agent.
5. **Define tools** — Write Python functions for each capability the agent needs. Test them independently.
6. **Assemble the agent** — Use LangGraph to wire the LLM, tools, and RAG together into an agent graph.
7. **Test iteratively** — Run the agent on sample queries. Inspect the agent's reasoning (which tools it called, in what order, what context it used). Adjust prompts, tools, and retrieval parameters.
8. **Add guardrails** — Input validation, output filtering, rate limits, human-in-the-loop approvals for sensitive actions.
9. **Deploy** — Package as a container, deploy to Code Engine or IKS, front with API Connect.
10. **Monitor** — Set up watsonx.governance for production monitoring of agent quality, fairness, and drift.

---

## References

### Concepts
- [ReAct: Synergizing Reasoning and Acting in Language Models (paper)](https://arxiv.org/abs/2210.03629)
- [LLM Powered Autonomous Agents (Lilian Weng)](https://lilianweng.github.io/posts/2023-06-23-agent/)

### IBM
- [watsonx Developer Hub](https://www.ibm.com/watsonx/developer/)
- [watsonx Agent Quickstart](https://www.ibm.com/watsonx/developer/agents/quickstart)
- [Agent Lab — Documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-agent-lab.html?context=wx)
- [watsonx Orchestrate — AI Agent Builder](https://www.ibm.com/products/watsonx-orchestrate/ai-agent-builder)
- [IBM Granite Models](https://www.ibm.com/granite)
- [Granite 4.0 Documentation](https://www.ibm.com/granite/docs/models/granite)
- [ibm_watsonx_ai Python SDK (v1.5.1)](https://ibm.github.io/watsonx-ai-python-sdk/v1.5.1/api.html)
- [langchain-ibm — PyPI](https://pypi.org/project/langchain-ibm/)
- [watsonx.ai Sample Notebooks (agents, RAG)](https://github.com/IBM/watsonx-ai-samples)

### LangChain / LangGraph
- [LangChain Documentation](https://python.langchain.com/docs/introduction/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph — How to Create a ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)
- [LangChain Tools — Docs](https://python.langchain.com/docs/how_to/#tools)

### RAG
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers Documentation](https://www.sbert.net/)

### Deployment
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [IBM Code Engine — Docs](https://cloud.ibm.com/docs/codeengine?topic=codeengine-about)
