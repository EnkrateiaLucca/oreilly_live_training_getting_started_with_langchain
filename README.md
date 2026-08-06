# Getting Started with LangChain 1.0+

[![LangChain](https://img.shields.io/badge/LangChain-1.0+-blue)](https://docs.langchain.com/)
[![Python](https://img.shields.io/badge/Python-3.11--3.13-yellow)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-green)](https://langchain-ai.github.io/langgraph/)

O'Reilly Live Training — Build AI agents with LangChain 1.0 and LangGraph.

[Live-training official website from O'Reilly](https://learning.oreilly.com/live-events/getting-started-with-langchain/0636920098586/0636920098585/)

## Requirements

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/) package manager
- [OpenAI API key](https://platform.openai.com/)
- [Tavily API key](https://tavily.com/) (for search tools)
- [Node.js](https://nodejs.org/) (for Agent Chat UI)
- [LangSmith account](https://smith.langchain.com/) (optional, for observability)

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain.git
cd oreilly_live_training_getting_started_with_langchain

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 3. Install dependencies
uv sync

# 4. Register the Jupyter kernel the notebooks expect
uv run python -m ipykernel install --user --name=oreilly-langchain --display-name "oreilly-langchain"

# 5. Launch Jupyter
uv run jupyter lab
```

When opening a notebook, select the **oreilly-langchain** kernel (Kernel → Change Kernel) if it isn't picked up automatically.

## Course Structure

### Notebook 0: Agents From Scratch
[`notebooks/00-langchain-basics-intro.ipynb`](notebooks/00-langchain-basics-intro.ipynb)

Builds intuition before reaching for the framework's abstractions:
- **Core components** — an LLM, a set of tools, and a loop that ties them together
- **An agent, from scratch** — manually bind tools to a chat model, detect `tool_calls`, execute them, and loop until the model is done
- **`create_agent`** — the same task solved in a few lines, so you can see exactly what the abstraction replaces

### Notebook 1: LangChain 1.0 Fundamentals
[`notebooks/1.0-langchain-fundamentals.ipynb`](notebooks/1.0-langchain-fundamentals.ipynb)

Covers the 6 essential building blocks:
- **Agents** — `create_agent()` API
- **Models** — `init_chat_model()` universal factory
- **Messages** — SystemMessage, HumanMessage, AIMessage, ToolMessage
- **Tools** — `@tool` decorator and dependency injection
- **Short-term Memory** — Checkpointers and thread isolation
- **Streaming** — Real-time output modes

### Notebook 2: Structured Outputs & Practical Applications
[`notebooks/2.0-structured-outputs.ipynb`](notebooks/2.0-structured-outputs.ipynb)

- **Structured Output Fundamentals** — ProviderStrategy, ToolStrategy, Pydantic schemas
- **Resume Analyzer & Job-Fit Scorer** — Real-world application with nested data models

### Demo: Deployable Chat-Over-Docs Agent
[`demo/`](demo/)

A complete RAG agent deployed with LangGraph, featuring:
- Document loading and vector search
- LangGraph deployment via `langgraph dev`
- Agent Chat UI for interactive testing
- LangSmith observability

### Demo: Local Sandboxed Agent (Ollama)
[`notebooks/local_agent.py`](notebooks/local_agent.py)

A fully local, self-contained agent (`uv run notebooks/local_agent.py`) that shows `create_agent` running against a local model instead of a hosted API:
- `init_chat_model("ollama:gemma4")` — no API key required
- File tools (`read_file`, `write_file`, `edit_file`, `delete_file`) and a `bash` tool, all sandboxed to `notebooks/agent_workspace/` with a path-escape guard and a blocklist for dangerous commands
- Web search via `TavilySearch`
- A system prompt that has the agent log a one-line summary of every session to `memory.md`

Requires [Ollama](https://ollama.com/) running locally with the `gemma4` model pulled (`ollama pull gemma4`).

### Demo: Chat-Over-PDFs, Deployed to Vercel
[`internal-docs-search-agent/`](internal-docs-search-agent/)

The same `create_agent` RAG pattern from the `demo/` app, taken all the way to a public web deployment on Vercel — a single-page chat UI next to a PDF viewer that jumps to the cited page:
- Vector index pre-built offline into a JSON file (`scripts/build_index.py`) since serverless can't host Chroma — the deployed function loads it and does cosine similarity with numpy
- One-file FastAPI app (`main.py`) serving the agent, the `/api/chat` endpoint, and the embedded chat/viewer UI
- Full architecture, deploy steps, and gotchas in [`internal-docs-search-agent/DEPLOYMENT_GUIDE.md`](internal-docs-search-agent/DEPLOYMENT_GUIDE.md)

This deployment is torn down between classes to avoid an idle public endpoint burning API credits — follow the deployment guide to bring it back up live.

## Live Demo

```bash
# Start the agent server
make demo
```

Then visit [agentchat.vercel.app](https://agentchat.vercel.app) and connect to `http://localhost:2024` with graph ID `chat_over_docs`.

## Previous Course Materials

- `deprecated/` — Pre-April 2026 course materials
- `archive/pre-v1/` — Pre-LangChain 1.0 materials

## Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/) — Tracing and debugging
- [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui)

## License

This project is licensed under the MIT License.
