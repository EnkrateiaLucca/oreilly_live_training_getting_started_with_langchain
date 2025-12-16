# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

O'Reilly Live Training course teaching LangChain 1.0+ and LangGraph for building AI applications. The course covers agents, RAG systems, and complex AI workflows.

## LangChain 1.0+ Compliance (Important)

**All code in this repository must follow LangChain >= 1.0.0 documentation and patterns.** When writing or modifying code, always use the latest LangChain 1.0 APIs. Reference the official documentation at https://docs.langchain.com/ for current patterns.

### Deprecated → Current API Migration

| Deprecated (Pre-v1) | Current (v1.0+) |
|---------------------|-----------------|
| `create_react_agent()` | `create_agent()` |
| `create_tool_calling_agent()` | `create_agent()` |
| `AgentExecutor` | `create_agent()` returns a runnable graph directly |
| `langchain.chat_models.ChatOpenAI` | `langchain_openai.ChatOpenAI` |
| `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` |
| `langchain_community.tools.tavily_search.TavilySearchResults` | `langchain_tavily.TavilySearch` |
| `langchain.hub` | `langchain_classic.hub` (only for legacy prompts) |
| `ConversationBufferMemory` | Use LangGraph checkpointing or message history |
| `LLMChain` | Use LCEL (`prompt | model | parser`) |
| `SequentialChain` | Use LCEL with `RunnableSequence` |

### Key v1.0 Changes to Remember

1. **Agents return graphs**: `create_agent()` returns a LangGraph `CompiledGraph`, not an executor
2. **Messages-based I/O**: Agents use `{"messages": [...]}` format, not `{"input": "..."}`
3. **Python 3.10+ required**: Python 3.9 is no longer supported
4. **Separate packages**: Use provider-specific packages (`langchain-openai`, `langchain-anthropic`, etc.)

## Development Commands

```bash
# Setup with Makefile (uses uv + conda)
make all                    # Full setup: conda env, pip-tools, notebook kernel

# Manual setup
conda create -n oreilly-langchain python=3.12
conda activate oreilly-langchain
pip install -r requirements/requirements.txt

# Jupyter kernel
python -m ipykernel install --user --name=oreilly-langchain

# Dependency management (uses uv pip-compile)
make env-update            # Compile and sync requirements
make freeze                # Freeze current environment
```

## Required Environment Variables

```bash
export OPENAI_API_KEY="..."
export TAVILY_API_KEY="..."           # For search tools
export LANGCHAIN_API_KEY="..."        # Optional: LangSmith tracing
```

## LangChain 1.0 Patterns (Critical)

This codebase uses LangChain 1.0+ patterns. Key differences from pre-v1:

### Agent Creation
```python
# LangChain 1.0 pattern (use this)
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o-mini",  # String format
    tools=tools,
    system_prompt="...",
)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

# NOT the old AgentExecutor pattern
```

### Tool Definition
```python
from langchain_core.tools import tool

@tool
def my_tool(param: str) -> str:
    """Tool docstring becomes the tool description."""
    return result
```

### Streaming
```python
# Token-by-token streaming
for token, metadata in agent.stream({"messages": "..."}, stream_mode="messages"):
    print(token.content, end="")

# State streaming
for step in agent.stream({"messages": "..."}, stream_mode="values"):
    step["messages"][-1].pretty_print()
```

### Runtime Context (Dependency Injection)
```python
from dataclasses import dataclass
from langgraph.runtime import get_runtime

@dataclass
class RuntimeContext:
    db: SQLDatabase

@tool
def execute_sql(query: str) -> str:
    runtime = get_runtime(RuntimeContext)
    return runtime.context.db.run(query)

agent = create_agent(..., context_schema=RuntimeContext)
agent.invoke({...}, context=RuntimeContext(db=db))
```

## Repository Structure

- `notebooks/` - Main course Jupyter notebooks (numbered by module)
- `scripts/` - Standalone Python examples (jira-agent.py, rag_methods.py)
- `archive/pre-v1/` - Legacy pre-LangChain-1.0 notebooks (reference only)
- `requirements/` - Dependencies managed via pip-tools (requirements.in → requirements.txt)

## Import Conventions

```python
# Models
from langchain_openai import ChatOpenAI

# Core
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# Agents
from langchain.agents import create_agent

# Search (LangChain 1.0)
from langchain_tavily import TavilySearch  # NOT langchain_community.tools.tavily_search
```
