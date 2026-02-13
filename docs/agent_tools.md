# Agent Tools — How LLMs Interact with the Outside World

This guide explains what agent tools are, how they work under the hood, and how to build them for IBM watsonx-powered agents. It is a companion to the [Building AI Agents](build_agents.md) guide, which covers the broader agent architecture.

**Table of Contents**

1. [Why Agent Tools Exist](#1-why-agent-tools-exist)
2. [What Is an Agent Tool?](#2-what-is-an-agent-tool)
3. [How Tool Calling Works Under the Hood](#3-how-tool-calling-works-under-the-hood)
   - 3.1 [What the LLM Actually Sees](#31-what-the-llm-actually-sees)
   - 3.2 [Multi-Step Tool Use](#32-multi-step-tool-use)
4. [Types of Agent Tools](#4-types-of-agent-tools)
   - 4.1 [Retrieval / Search Tools](#41-retrieval--search-tools)
   - 4.2 [Database / SQL Tools](#42-database--sql-tools)
   - 4.3 [API Integration Tools](#43-api-integration-tools)
   - 4.4 [Code Execution Tools](#44-code-execution-tools)
   - 4.5 [Custom Tools](#45-custom-tools)
5. [How IBM watsonx Models Support Tool Use](#5-how-ibm-watsonx-models-support-tool-use)
6. [Building Tools in Practice](#6-building-tools-in-practice)
   - 6.1 [Defining a Tool with `@tool`](#61-defining-a-tool-with-tool)
   - 6.2 [Using a Toolkit](#62-using-a-toolkit)
   - 6.3 [Wiring Tools into an Agent](#63-wiring-tools-into-an-agent)
7. [The Tool Calling Lifecycle (End-to-End)](#7-the-tool-calling-lifecycle-end-to-end)
8. [Why Tools Make Agents Useful](#8-why-tools-make-agents-useful)
9. [Best Practices for Designing Tools](#9-best-practices-for-designing-tools)
10. [Tools in This Project](#10-tools-in-this-project)
- [References](#references)

---

## 1. Why Agent Tools Exist

A large language model is trained on a static snapshot of text. Once training ends, the model has no way to:

- Look up today's stock price or weather
- Query your company's internal database
- Search a private knowledge base
- Send an email or create a calendar event
- Execute code or call an API

Think of it this way: an LLM is like an expert analyst locked in a room with no phone, no computer, and no filing cabinet. The analyst can reason brilliantly about whatever is written on the whiteboard in front of them, but they cannot *do* anything or *learn* anything new.

**Tools are the phone, computer, and filing cabinet.** They give the LLM a way to reach outside its context window and interact with the real world.

Consider a concrete example. A user asks:

> "What was our total revenue in Q3 2025?"

Without tools, the LLM can only guess or refuse. With a database query tool, the agent can:

1. Recognize that it needs to look up financial data
2. Call the database tool with an appropriate SQL query
3. Read the result
4. Formulate a grounded answer

This is the core idea behind the Think-Act-Observe loop described in [build_agents.md](build_agents.md) Section 1 — tools are what make the "Act" step possible.

---

## 2. What Is an Agent Tool?

An **agent tool** is a callable function that an LLM can invoke to perform an action or retrieve information that lies outside its training data. Every tool has four components:

| Component | Purpose | Example |
|---|---|---|
| **Name** | Unique identifier the LLM uses to select the tool | `search_knowledge_base` |
| **Description** | Natural-language explanation of what the tool does and when to use it — this is what the LLM reads to decide whether to call it | "Search the knowledge base for information relevant to the query. Use this tool when you need to find information from indexed documents." |
| **Input schema** | The parameters the tool accepts (names, types, descriptions), usually expressed as JSON Schema or a Pydantic model | `{ "query": { "type": "string", "description": "The search query..." } }` |
| **Implementation** | The Python function that actually executes when the tool is called | A function that queries a vector database and returns formatted results |

The description is especially important. The LLM has no access to the implementation — it decides which tool to call based entirely on the name, description, and parameter schema. A vague or misleading description leads to incorrect tool selection.

Here is a real tool definition from this project (`app/tools/search.py`):

```python
from langchain_core.tools import tool

from app.prompts.templates import format_retrieved_context
from app.rag.retrieval import retrieve_documents


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information relevant to the query.

    Use this tool when you need to find information from indexed documents
    to answer the user's question.

    Parameters
    ----------
    query:
        The search query describing what information is needed.
    """
    docs = retrieve_documents(query)
    return format_retrieved_context(docs)
```

Notice three things:

1. The `@tool` decorator converts a plain Python function into a LangChain tool
2. The **docstring** becomes the tool description — the LLM reads this text to decide when to call the tool
3. The **type annotations** (`query: str`) define the input schema automatically

See also [build_agents.md](build_agents.md) Section 2.2 for the broader context of tools within an agent system.

---

## 3. How Tool Calling Works Under the Hood

Tool calling is not magic — it is a structured output capability built into modern LLMs. When a model supports tool calling, it can choose to emit a structured tool invocation (typically JSON) instead of plain text, when it determines that a tool would help answer the question.

Here is the lifecycle of a single tool call:

```
User: "What is our refund policy for items over $100?"
  │
  ▼
┌────────────────────────────────────────────────────────────────┐
│ LLM receives: system prompt + tool definitions + user message  │
│                                                                │
│ LLM reasons: "I need to search the knowledge base for the     │
│               refund policy."                                  │
│                                                                │
│ LLM outputs (structured):                                     │
│   tool_call: search_knowledge_base(query="refund policy       │
│              items over $100")                                 │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ Agent runtime executes the function:                           │
│   search_knowledge_base("refund policy items over $100")      │
│                                                                │
│ Returns: "[1] (source: handbook.pdf)\nItems over $100 may be  │
│           returned within 30 days for a full refund..."        │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ LLM receives the tool result as a new message in the          │
│ conversation history                                          │
│                                                                │
│ LLM reasons: "I have the information I need."                 │
│                                                                │
│ LLM outputs (plain text):                                     │
│   "Our refund policy for items over $100 allows returns       │
│    within 30 days for a full refund..."                        │
└────────────────────────────────────────────────────────────────┘
```

### 3.1 What the LLM Actually Sees

When tools are registered with an agent, their definitions are serialized and included in the system prompt (or a dedicated tools section of the API request). The LLM sees something like this:

```json
{
  "name": "search_knowledge_base",
  "description": "Search the knowledge base for information relevant to the query. Use this tool when you need to find information from indexed documents to answer the user's question.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "The search query describing what information is needed."
      }
    },
    "required": ["query"]
  }
}
```

The LLM uses this definition — not the Python source code — to decide when and how to invoke the tool. This is why descriptive docstrings and clear parameter names matter so much.

### 3.2 Multi-Step Tool Use

An agent often needs multiple tool calls to answer a single question. This is the **ReAct pattern** (Reasoning + Acting) described in [build_agents.md](build_agents.md) Section 1:

1. **Think** — The LLM reasons about the current state and decides what to do next
2. **Act** — The LLM emits a tool call
3. **Observe** — The agent runtime executes the tool and feeds the result back to the LLM
4. **Repeat** — The LLM decides whether to call another tool or produce a final answer

For example, answering "Compare our Q2 and Q3 revenue" might require two sequential database queries — one for Q2 and one for Q3 — before the LLM can compute the comparison and respond.

---

## 4. Types of Agent Tools

Tools can be organized by the kind of external system they interact with:

| Category | What It Does | Example |
|---|---|---|
| **Retrieval / Search** | Query a vector database or search engine for relevant documents | `search_knowledge_base` |
| **Database / SQL** | Execute SQL queries against relational or lakehouse databases | `sql_db_query`, `sql_db_list_tables` |
| **API Integration** | Call external REST APIs (weather, CRM, payment, etc.) | `get_weather`, `create_ticket` |
| **Code Execution** | Run Python code or shell commands in a sandbox | `python_repl` |
| **File Operations** | Read, write, or parse files (PDFs, CSVs, spreadsheets) | `read_pdf`, `parse_csv` |
| **Communication** | Send emails, post messages, create calendar events | `send_email`, `post_slack_message` |

### 4.1 Retrieval / Search Tools

Retrieval tools connect the agent to a knowledge base — a collection of documents that have been embedded and stored in a vector database. The typical pattern is a two-stage pipeline: **retrieve** candidates via vector similarity search, then **rerank** them with a cross-encoder model for higher precision.

This project implements this pattern across two files. The retrieval pipeline (`app/rag/retrieval.py`):

```python
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_ibm import WatsonxRerank

from app.core.client import get_watsonx_client
from app.rag.vectorstore import get_vectorstore


def retrieve_documents(query, collection_name=None, k=4, rerank=True, initial_k=20):
    """Embed the query and return the top-k most relevant documents.

    When rerank is True, the pipeline first retrieves initial_k candidates
    from Milvus, then reranks them with WatsonxRerank and returns the top k.
    """
    vectorstore = get_vectorstore(collection_name)

    if not rerank:
        return vectorstore.similarity_search(query, k=k)

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": initial_k})
    compressor = WatsonxRerank(
        model_id="cross-encoder/ms-marco-minilm-l-12-v2",
        watsonx_client=get_watsonx_client(),
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
    return compression_retriever.invoke(query)[:k]
```

The tool wrapper (`app/tools/search.py`) calls this pipeline and formats the output:

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information relevant to the query.

    Use this tool when you need to find information from indexed documents
    to answer the user's question.
    """
    docs = retrieve_documents(query)
    return format_retrieved_context(docs)
```

The formatting step (`app/prompts/templates.py`) ensures the LLM receives a concise, structured context string rather than raw document objects:

```python
def format_retrieved_context(documents: list) -> str:
    if not documents:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)
```

For background on the Milvus vector database integration, see [ibm_watsonx.md](ibm_watsonx.md) Section 2 (watsonx.data).

### 4.2 Database / SQL Tools

Database tools let the agent query structured data using SQL. Instead of defining a single tool, the common pattern is a **toolkit** — a factory that generates a coordinated set of tools from a database connection.

This project uses `WatsonxSQLDatabaseToolkit` from `langchain-ibm`, which generates four tools:

| Tool | Purpose |
|---|---|
| `sql_db_list_tables` | List available tables in the database |
| `sql_db_schema` | Get the schema (columns, types) for specific tables |
| `sql_db_query` | Execute a SQL query and return results |
| `sql_db_query_checker` | Validate a SQL query before execution (uses the LLM) |

The toolkit factory from this project (`app/tools/database.py`):

```python
from langchain_ibm import WatsonxSQLDatabase, WatsonxSQLDatabaseToolkit

from app.core.client import get_watsonx_client
from app.core.llm import get_llm


def get_database_tools(connection_id, schema=None, llm=None):
    """Return watsonx SQL tools (query, info, list tables, query checker)."""
    db = WatsonxSQLDatabase(
        connection_id=connection_id,
        watsonx_client=get_watsonx_client(),
    )
    toolkit = WatsonxSQLDatabaseToolkit(db=db, llm=llm or get_llm())
    return toolkit.get_tools()
```

The agent uses these tools in sequence: it lists tables, inspects schemas, constructs a query, checks it, and then executes it — all driven by the LLM's reasoning, not hard-coded logic.

For background on watsonx.data, see [ibm_watsonx.md](ibm_watsonx.md) Section 2.

### 4.3 API Integration Tools

Any external API can be wrapped as a tool. Here is a pseudo-code example:

```python
import httpx
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Parameters
    ----------
    city:
        The city name (e.g. "Toronto", "New York").
    """
    response = httpx.get(
        "https://api.weather.example.com/current",
        params={"city": city},
    )
    data = response.json()
    return f"{data['city']}: {data['temperature']}°C, {data['condition']}"
```

The key principle is the same: wrap the API call in a function, add a descriptive docstring, and apply `@tool`.

### 4.4 Code Execution Tools

Code execution tools allow the agent to run Python code in a sandboxed environment. This is useful for calculations, data transformations, or generating visualizations. LangChain provides `PythonREPLTool` for this purpose. These tools require careful sandboxing to prevent arbitrary code execution in production environments.

### 4.5 Custom Tools

Any Python function can become a tool. If there is a programmatic way to accomplish a task, you can wrap it with `@tool`:

```python
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies using current exchange rates.

    Parameters
    ----------
    amount:
        The monetary amount to convert.
    from_currency:
        The source currency code (e.g. "USD", "EUR").
    to_currency:
        The target currency code.
    """
    rate = get_exchange_rate(from_currency, to_currency)
    converted = amount * rate
    return f"{amount} {from_currency} = {converted:.2f} {to_currency}"
```

---

## 5. How IBM watsonx Models Support Tool Use

Tool calling requires model-level support — the model must be able to generate structured tool invocations rather than free-form text. IBM Granite models have native function-calling support.

### Model Capabilities

| Model | Function Calling | Notes |
|---|---|---|
| **IBM Granite 4.0** (Micro, Tiny, Small H) | Native | Industry-leading BFCLv3 scores. Recommended for agent workloads. |
| **IBM Granite 3.x** (8B Instruct, etc.) | Native | Solid function-calling support. |
| **Meta Llama 3.3 70B Instruct** | Native | Strong function-calling via watsonx.ai hosting. |
| **Mistral Large 3** | Native | 256k context window, strong tool use. |

See [ibm_watsonx.md](ibm_watsonx.md) Section 1 (Model Catalog) for the full model list.

### `ChatWatsonx` and `bind_tools()`

The LangChain integration layer (`langchain-ibm`) provides `ChatWatsonx`, which supports tool calling through the standard LangChain interface. There are two ways to give a model access to tools:

**Explicit `bind_tools()`** — manually attach tool definitions to the model:

```python
from langchain_ibm import ChatWatsonx
from app.tools.search import search_knowledge_base

llm = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    watsonx_client=client,
    params={"max_new_tokens": 4096, "temperature": 0.7},
)

# Bind tool definitions so the model knows they are available
llm_with_tools = llm.bind_tools([search_knowledge_base])
```

After `bind_tools()`, the model's API requests include the serialized tool definitions, and the model can choose to emit tool calls in its responses.

**LangGraph `create_react_agent()`** — handles tool binding automatically:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=[search_knowledge_base],
)
```

When you use `create_react_agent`, you do not need to call `bind_tools()` yourself — LangGraph binds the tools to the model internally and manages the full ReAct loop (call LLM, execute tool, feed result back, repeat).

This project uses the `create_react_agent` approach in `app/agents/rag_agent.py`.

---

## 6. Building Tools in Practice

### 6.1 Defining a Tool with `@tool`

The `@tool` decorator from `langchain_core.tools` is the simplest way to create a tool. Here are three examples of increasing complexity.

**Single parameter:**

```python
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic."""
    # implementation
    return results
```

**Multiple parameters:**

```python
@tool
def query_sales(region: str, quarter: str, year: int) -> str:
    """Query the sales database for revenue by region and time period.

    Parameters
    ----------
    region:
        Geographic region (e.g. "EMEA", "North America", "APAC").
    quarter:
        Fiscal quarter (e.g. "Q1", "Q2", "Q3", "Q4").
    year:
        Fiscal year (e.g. 2025).
    """
    # implementation
    return results
```

**Pydantic input schema** (for complex or validated inputs):

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class FilterCriteria(BaseModel):
    """Criteria for filtering documents."""
    query: str = Field(description="The search query text")
    doc_type: str = Field(description="Document type: 'policy', 'procedure', or 'faq'")
    max_results: int = Field(default=5, description="Maximum number of results to return")


@tool(args_schema=FilterCriteria)
def filtered_search(query: str, doc_type: str, max_results: int = 5) -> str:
    """Search the knowledge base with document type filtering."""
    # implementation
    return results
```

In all cases, the docstring is the tool description, and the type annotations (or Pydantic schema) define the input schema.

### 6.2 Using a Toolkit

When a tool category requires multiple coordinated tools (like database operations), a **toolkit** pattern is cleaner than defining each tool individually. A toolkit is a factory that inspects a resource (such as a database connection) and generates a set of tools automatically.

From this project (`app/tools/database.py`):

```python
from langchain_ibm import WatsonxSQLDatabaseToolkit

db = get_sql_database(connection_id)
toolkit = WatsonxSQLDatabaseToolkit(db=db, llm=get_llm())
tools = toolkit.get_tools()  # Returns 4 tools: list, schema, query, checker
```

The toolkit handles tool naming, descriptions, and schemas — you just provide the database connection and an LLM (used by the query-checker tool to validate SQL before execution).

### 6.3 Wiring Tools into an Agent

Once tools are defined, they are assembled into an agent. From this project (`app/agents/rag_agent.py`):

```python
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from app.core.llm import get_chat_model
from app.prompts.templates import RAG_SYSTEM_PROMPT
from app.tools.search import search_knowledge_base


def build_agent(db_tools: list[BaseTool] | None = None):
    """Build and return a compiled LangGraph ReAct agent."""
    llm = get_chat_model()
    tools: list[BaseTool] = [search_knowledge_base]
    if db_tools:
        tools.extend(db_tools)

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=RAG_SYSTEM_PROMPT,
    )
    return agent
```

This function:

1. Creates a `ChatWatsonx` model instance
2. Assembles the tool list (always includes `search_knowledge_base`, optionally adds database tools)
3. Passes tools + model + system prompt to `create_react_agent`, which builds the full ReAct graph

See [build_agents.md](build_agents.md) Section 5 for the step-by-step agent assembly walkthrough.

---

## 7. The Tool Calling Lifecycle (End-to-End)

Here is the complete lifecycle showing how the agent runtime mediates between the user, the LLM, and the tools:

```
┌──────┐       ┌──────────────┐       ┌───────────────┐       ┌───────────┐
│ User │       │ Agent Runtime│       │     LLM       │       │   Tools   │
└──┬───┘       └──────┬───────┘       └───────┬───────┘       └─────┬─────┘
   │                  │                       │                     │
   │  1. Send query   │                       │                     │
   │────────────────▶│                       │                    │
   │                  │  2. Forward query +   │                    │
   │                  │     tool definitions  │                    │
   │                  │─────────────────────▶│                    │
   │                  │                       │                    │
   │                  │  3. LLM returns       │                    │
   │                  │     tool_call(...)    │                    │
   │                  │◀─────────────────────│                    │
   │                  │                       │                    │
   │                  │  4. Execute tool      │                    │
   │                  │───────────────────────────────────────────▶│
   │                  │                      │                    │
   │                  │  4b. Return result    │                    │
   │                  │◀──────────────────────────────────────────│
   │                  │                      │                    │
   │                  │  5. Feed result back  │                    │
   │                  │     to LLM           │                    │
   │                  │─────────────────────▶│                    │
   │                  │                      │                    │
   │                  │     (steps 3-5 may    │                    │
   │                  │      repeat if the    │                    │
   │                  │      LLM needs more   │                    │
   │                  │      tool calls)      │                    │
   │                  │                       │                    │
   │                  │  6. LLM returns       │                    │
   │                  │     final text answer │                    │
   │                  │◀─────────────────────│                    │
   │                  │                      │                    │
   │  7. Return answer│                      │                    │
   │◀────────────────│                      │                    │
```

**Step-by-step:**

1. **User sends a query** — e.g., "What is our refund policy for items over $100?"
2. **Agent runtime forwards the query to the LLM** — along with the system prompt and serialized tool definitions
3. **LLM reasons and emits a tool call** — instead of text, the model outputs a structured invocation like `search_knowledge_base(query="refund policy items over $100")`
4. **Agent runtime executes the tool** — calls the actual Python function with the specified arguments, receives the return value
5. **Agent runtime feeds the tool result back to the LLM** — as a new message in the conversation, so the LLM can incorporate the information
6. **(Optional repeat)** — The LLM may decide it needs another tool call (steps 3-5 repeat). For example, it might search the knowledge base and then query a database for additional details.
7. **LLM produces a final text answer** — once it has enough information, it responds with plain text that the agent runtime returns to the user

This is the ReAct loop in action. See [build_agents.md](build_agents.md) Section 1 for the conceptual diagram of the Think-Act-Observe pattern.

---

## 8. Why Tools Make Agents Useful

Without tools, an LLM is limited to what it learned during training. With tools, it becomes a system that can act on the world. Here is the contrast:

| Dimension | Without Tools | With Tools |
|---|---|---|
| **Data freshness** | Limited to training data cutoff | Can query live databases, APIs, and knowledge bases |
| **Accuracy on specific facts** | May hallucinate details it does not know | Can retrieve and cite authoritative sources |
| **Actionability** | Can only suggest actions ("you should send an email") | Can perform actions (actually send the email) |
| **Specialization** | General knowledge only | Can access domain-specific systems (ERP, CRM, databases) |
| **Auditability** | Opaque reasoning — hard to verify claims | Tool calls create a traceable log of what data was accessed and what actions were taken |

Auditability is especially important in enterprise settings. When an agent's answer includes a citation like `[1] (source: handbook.pdf)`, a human reviewer can verify the claim. watsonx.governance can monitor tool usage patterns and flag anomalies. See [ibm_watsonx.md](ibm_watsonx.md) Section 3 (watsonx.governance) for details on agent monitoring.

---

## 9. Best Practices for Designing Tools

1. **Write descriptive docstrings.** The LLM chooses tools based on the description. Include what the tool does, when to use it, and what it returns. Be specific — "Search the knowledge base for policy documents" is better than "Search stuff."

2. **One tool, one job.** Each tool should do one thing well. A tool that searches the knowledge base should not also send emails. If the agent needs both capabilities, define two tools.

3. **Use clear parameter names and descriptions.** `query` is better than `q`. Add descriptions to parameters so the LLM knows what values to pass.

4. **Return concise, structured output.** The tool result goes back into the LLM's context window. Large outputs waste tokens and may confuse the model. Format results for readability — this project's `format_retrieved_context()` in `app/prompts/templates.py` is a good example:

   ```python
   # Returns structured, numbered excerpts with source attribution
   # "[1] (source: handbook.pdf)\nItems over $100 may be returned..."
   ```

5. **Handle errors gracefully.** Return a descriptive error string rather than raising an exception. The LLM can reason about "Error: database connection failed" and try an alternative approach; an unhandled exception terminates the agent loop.

6. **Validate inputs.** Check parameter types and ranges before executing. Return a clear message if inputs are invalid, so the LLM can correct its call.

7. **Test tools independently.** Each tool should work correctly as a standalone function before being wired into an agent. This makes debugging much easier — if the agent gives a wrong answer, you can isolate whether the problem is in the tool or the LLM's reasoning.

8. **Limit the number of tools.** More tools means more text in the system prompt and more choices for the LLM to reason about. If an agent has 20 tools, the model is more likely to pick the wrong one. Group related functionality into toolkits and only expose tools the agent actually needs.

9. **Consider idempotency and safety.** Read-only tools (search, query) are safe to retry. Write tools (send email, delete record) should be designed carefully — consider confirmation steps or human-in-the-loop approval for irreversible actions.

---

## 10. Tools in This Project

This project defines the following tools:

| Tool | File | Description |
|---|---|---|
| `search_knowledge_base` | `app/tools/search.py` | Searches the Milvus vector database for documents relevant to a query, using a two-stage retrieve + rerank pipeline |
| SQL toolkit (4 tools) | `app/tools/database.py` | `sql_db_list_tables`, `sql_db_schema`, `sql_db_query`, `sql_db_query_checker` — generated by `WatsonxSQLDatabaseToolkit` for querying watsonx.data |

The **retrieval pipeline** works as follows:

1. The user's query is embedded using `WatsonxEmbeddings` (configured in `app/core/embeddings.py`)
2. The embedding is used to search a Milvus collection for the 20 nearest vectors (`app/rag/vectorstore.py`)
3. The 20 candidates are reranked by `WatsonxRerank` using a cross-encoder model (`app/rag/retrieval.py`)
4. The top 4 results are formatted into a numbered context string (`app/prompts/templates.py`)
5. The formatted string is returned to the agent as the tool result

The **database toolkit** connects to watsonx.data via Arrow Flight and generates four tools that let the agent explore and query structured data without hard-coded SQL.

Both tool types are assembled into a single agent by `build_agent()` in `app/agents/rag_agent.py`, which passes them to LangGraph's `create_react_agent`.

For infrastructure details, see [ibm_watsonx.md](ibm_watsonx.md) (watsonx.ai and watsonx.data) and [ibm_cloud.md](ibm_cloud.md) (IBM Cloud Object Storage and deployment).

---

## References

### Concepts

- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)

### IBM

- [IBM Granite Models](https://www.ibm.com/granite)
- [Granite 4.0 Documentation](https://www.ibm.com/granite/docs/models/granite)
- [watsonx Developer Hub](https://www.ibm.com/watsonx/developer/)
- [watsonx Agent Quickstart](https://www.ibm.com/watsonx/developer/agents/quickstart)
- [ibm_watsonx_ai Python SDK (v1.5.1)](https://ibm.github.io/watsonx-ai-python-sdk/v1.5.1/api.html)
- [langchain-ibm — PyPI](https://pypi.org/project/langchain-ibm/)

### LangChain / LangGraph

- [LangChain Tools — Documentation](https://python.langchain.com/docs/how_to/#tools)
- [LangChain — `@tool` Decorator](https://python.langchain.com/docs/how_to/custom_tools/#tool-decorator)
- [LangChain — StructuredTool](https://python.langchain.com/docs/how_to/custom_tools/#structuredtool)
- [LangGraph — How to Create a ReAct Agent](https://langchain-ai.github.io/langgraph/how-tos/create-react-agent/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

### This Project

- [Building AI Agents — A Practical Guide](build_agents.md)
- [IBM watsonx — The AI and Data Platform](ibm_watsonx.md)
- [IBM Cloud — Infrastructure and Services](ibm_cloud.md)
