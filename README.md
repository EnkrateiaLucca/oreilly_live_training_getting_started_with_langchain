# Getting Started with LangChain

[![LangChain](https://img.shields.io/badge/LangChain-1.0+-blue)](https://docs.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-green)](https://langchain-ai.github.io/langgraph/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://www.python.org/)

O'Reilly Live Training Course - Getting Started with LangChain

**Updated for LangChain 1.0+ (October 2025)**

[Live-training official website from O'Reilly](https://learning.oreilly.com/live-events/getting-started-with-langchain/0636920098586/0636920098585/)

## Course Overview

This course teaches you how to build AI applications using LangChain and LangGraph, the leading frameworks for LLM application development. You'll learn to create agents, build RAG systems, and orchestrate complex AI workflows.

## Requirements

- Python 3.10+ (Python 3.9 is no longer supported in LangChain 1.0)
- OpenAI API key
- Tavily API key (for search tools)

## Setup

### Using Conda

1. Install [Anaconda](https://www.anaconda.com/download)
2. Create an environment: `conda create -n oreilly-langchain python=3.11`
3. Activate your environment: `conda activate oreilly-langchain`
4. Install requirements: `pip install -r requirements/requirements.txt`
5. Setup your OpenAI [API key](https://platform.openai.com/)

### Using Pip

1. **Create a Virtual Environment:**
   ```bash
   python -m venv oreilly_env
   ```

2. **Activate the Virtual Environment:**
   - On macOS/Linux: `source oreilly_env/bin/activate`
   - On Windows: `.\oreilly_env\Scripts\activate`

3. **Install Dependencies:**
   ```bash
   pip install -r requirements/requirements.txt
   ```

4. **Setup Environment Variables:**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export TAVILY_API_KEY="your-tavily-key"
   export LANGCHAIN_API_KEY="your-langsmith-key"  # Optional: for tracing
   ```

### Using Makefile (with uv)

1. Install `uv` with `pip install uv`
2. Run `make all`

## Course Structure

### Module 1: Introduction to LangChain 1.0
**Notebook**: [`notebooks/1.0-intro-to-langchain.ipynb`](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/1.0-intro-to-langchain.ipynb)
- LangChain 1.0 architecture overview
- Chat models and message types
- Model initialization patterns
- LCEL (LangChain Expression Language)

### Module 2: Runnable Interface Deep Dive
**Notebook**: [`notebooks/1.1-intro-to-runnable-interface.ipynb`](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/1.1-intro-to-runnable-interface.ipynb)
- RunnableLambda, RunnableSequence, RunnableParallel
- RunnablePassthrough for data manipulation
- Building complex chains

### Module 3: Building Agents with LangChain 1.0
**Notebook**: [`notebooks/3.0-building-llm-agents-with-langchain.ipynb`](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/3.0-building-llm-agents-with-langchain.ipynb)
- The new `create_agent` API (replaces AgentExecutor)
- Tool definition with `@tool` decorator
- Structured output with agents
- Streaming agent responses
- Memory and checkpointing

### Module 4: LangGraph Fundamentals
**Notebook**: [`notebooks/4.0-langgraph-quick-introduction.ipynb`](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/4.0-langgraph-quick-introduction.ipynb)
- StateGraph basics
- Nodes, edges, and conditional routing
- Building stateful applications
- Checkpointing and memory

### Module 5: RAG with LangChain and LangGraph
**Notebook**: [`notebooks/2.0-simple-rag-langchain-langgraph.ipynb`](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/2.0-simple-rag-langchain-langgraph.ipynb)
- Document loading and splitting
- Vector stores and embeddings
- Retrieval chains
- Conversational RAG

### Module 6: Agents with LangGraph
**Notebook**: [`notebooks/3.2-langchain-agent-with-langgraph.ipynb`](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/3.2-langchain-agent-with-langgraph.ipynb)
- Custom agent workflows
- Tool integration
- When to use `create_agent` vs custom StateGraph

### Supplementary Materials
- [`notebooks/4.0-structured-research-report-generation.ipynb`](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/4.0-structured-research-report-generation.ipynb) - Structured output patterns
- [`notebooks/5.0-demos-research-workflows.ipynb`](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/5.0-demos-research-workflows.ipynb) - Research workflow examples
- [`notebooks/intro-rag-basics.ipynb`](https://colab.research.google.com/github/EnkrateiaLucca/oreilly_live_training_getting_started_with_langchain/blob/main/notebooks/intro-rag-basics.ipynb) - RAG fundamentals

## Key Changes in LangChain 1.0

### Agent API
```python
# OLD (deprecated)
from langchain.agents import create_tool_calling_agent, AgentExecutor
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "..."})

# NEW (LangChain 1.0)
from langchain.agents import create_agent
agent = create_agent(
    model="openai:gpt-4o",  # String format
    tools=tools,
    prompt="You are a helpful assistant.",
)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

### Search Tools
```python
# OLD (deprecated)
from langchain_community.tools.tavily_search import TavilySearchResults

# NEW (LangChain 1.0)
from langchain_tavily import TavilySearch
search = TavilySearch(max_results=3)
```

### Import Changes
| Old Import | New Import |
|------------|------------|
| `langchain.chat_models.ChatOpenAI` | `langchain_openai.ChatOpenAI` |
| `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` |
| `langchain.hub` | `langchain_classic.hub` (for legacy prompts) |

## Scripts

- `scripts/jira-agent.py` - Jira management agent using LangChain 1.0
- `scripts/rag_methods.py` - RAG utilities for Streamlit apps
- `scripts/langchain-app.py` - LangServe example
- `scripts/langchain-structured-output-ui.py` - Structured output demo

## Jupyter Kernel Setup

To use this environment with Jupyter Notebooks:
```bash
python -m ipykernel install --user --name=oreilly-langchain
```

## Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain 1.0 Release Blog](https://blog.langchain.com/langchain-langgraph-1dot0/)
- [LangSmith](https://smith.langchain.com/) - For tracing and debugging

## Archived Content

Legacy notebooks from pre-v1.0 courses have been moved to `archive/pre-v1/`.

## License

This project is licensed under the MIT License.
