# IBM watsonx Playground

# 1. Introduction

This repository is a learning and development environment for building AI agents on IBM's watsonx platform. It contains reference documentation, a ready-to-run agent skeleton, and automation to get a working setup quickly.

## What's Inside

### Documentation (`docs/`)

- **[IBM watsonx](docs/ibm_watsonx.md)** — Deep dive into the three watsonx pillars: watsonx.ai (foundation models, Prompt Lab, Tuning Studio), watsonx.data (lakehouse, Milvus vector DB), and watsonx.governance (fairness, drift, explainability).
- **[IBM Cloud](docs/ibm_cloud.md)** — Overview of IBM Cloud services relevant to AI agent workloads: Compute (VPC, IKS, Code Engine), Storage (COS), Networking & Security, Data & AI, and Integration.
- **[Building AI Agents](docs/build_agents.md)** — Practical guide covering agent components (LLM, tools, RAG, memory, orchestration), architectures (ReAct, RAG agent, multi-agent, human-in-the-loop), and a step-by-step LangGraph assembly walkthrough.
- **[Agent Tools](docs/agent_tools.md)** — How LLM tool calling works, types of tools (retrieval, database, API, code execution), building tools with LangChain's `@tool` decorator and toolkits, and the tool calling lifecycle.
- **[Git Workflow](docs/git_workflow.md)** — Step-by-step guide to the branching model, naming conventions, PR process, and conflict resolution used in this project.

### Agent Skeleton (`watsonx-agent-skeletton/`)

A working starter application that wires together the core technologies used throughout this project: IBM watsonx.ai for LLM inference, LangGraph for agent orchestration (might be replaced with watsonx Orchestrate), Milvus for vector search, IBM COS for document storage, and FastAPI for the web API. Its purpose is to give a quick overview of these technologies and provide a ready-made project setup to build on. See the [watsonx-agent-skeletton README](watsonx-agent-skeletton/README.md) for details.

# 2. Setup

The project uses Python 3.12.3 managed through [pyenv](https://github.com/pyenv/pyenv), with all dependencies isolated in a virtual environment. The setup automation handles installing pyenv, the correct Python version, creating the venv, installing packages, and preparing the `.env` credentials file — so you can go from a fresh clone to a running project with a single command.

## Prerequisites

| OS | Requirement |
|----|-------------|
| Ubuntu/Debian | `sudo apt install make` (usually pre-installed) |
| macOS | `xcode-select --install` (provides `make` and `git`) |
| Windows | PowerShell 5.1+ (pre-installed on Windows 10/11) |

## Quick Start

### Linux / macOS

```bash
make setup
```

This single command installs system dependencies, pyenv, Python 3.12.3, creates a virtual environment, installs all Python packages, and sets up your `.env` file.

### Windows

```powershell
.\setup.ps1
```

### After Setup

1. Edit `watsonx-agent-skeletton/config/.env` and fill in your IBM watsonx credentials and other environment variables.

## Make Targets

| Target | Description |
|--------|-------------|
| `make help` | Show available targets |
| `make setup` | One-shot full project setup |
| `make install-deps` | Install OS-level build dependencies |
| `make install-pyenv` | Install pyenv (skipped if already present) |
| `make install-python` | Install Python 3.12.3 via pyenv |
| `make venv` | Create virtual environment |
| `make install` | Install Python dependencies into venv |
| `make env` | Create `.env` from template |
| `make run` | Run the watsonx agent |
| `make clean` | Remove virtual environment |

## Adding Dependencies

If you install a new package, update the requirements file:

```bash
watsonx-agent-skeletton/venv/bin/pip install <package>
watsonx-agent-skeletton/venv/bin/pip freeze > watsonx-agent-skeletton/requirements.txt
```

<details>
<summary><strong>Manual Setup (without Make)</strong></summary>

If `make setup` did not work, follow these steps manually:

1. **Install pyenv**: https://github.com/pyenv/pyenv#installation
2. **Install Python**:
   ```bash
   pyenv install 3.12.3
   pyenv local 3.12.3
   ```
3. **Create venv and install dependencies**:
   ```bash
   python -m venv watsonx-agent-skeletton/venv
   watsonx-agent-skeletton/venv/bin/pip install -r watsonx-agent-skeletton/requirements.txt
   ```
4. **Set up credentials**:
   ```bash
   cp watsonx-agent-skeletton/config/.env.example watsonx-agent-skeletton/config/.env
   # Edit .env and fill in your values
   ```

</details>

# 3. Contributing

For the full step-by-step guide with commands, see **[docs/git_workflow.md](docs/git_workflow.md)**.

## Quick Rules

1. **Branch from `dev`**, not `main`. Create a `feature/*` or `fix/*` branch for your work.
2. **Never push directly to `main` or `dev`.** Always use a pull request.
3. **Prefix your commit messages** with `feat:`, `fix:`, or `docs:`.
4. **Open PRs into `dev`** and assign one teammate as reviewer.
5. **Merge your own PR** after it is approved, then delete the branch.

## Branch Naming

| Type | Pattern | Example |
|---|---|---|
| New feature | `feature/<short-description>` | `feature/add-auth` |
| Bug fix | `fix/<short-description>` | `fix/config-validation` |

Lowercase, hyphens between words, 2-4 words.
