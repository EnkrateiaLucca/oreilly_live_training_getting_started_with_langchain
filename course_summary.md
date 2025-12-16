# LangChain V1 Essentials - Course Summary

A compressed overview of the 9-notebook course covering LangChain/LangGraph agent fundamentals.

---

## L1: Fast Agent - Build a SQL Agent Fast

**Goal:** Rapidly build a functional SQL agent using `create_agent`.

**Key Concepts:**
- `create_agent(model, tools, system_prompt, context_schema)` - one-liner agent creation
- **RuntimeContext** - dataclass for dependency injection (e.g., database connection)
- `get_runtime()` - access context inside tools
- **ReAct Loop** - agent reasons, calls tools, observes results, repeats

**Pattern:**
```python
@dataclass
class RuntimeContext:
    db: SQLDatabase

@tool
def execute_sql(query: str) -> str:
    runtime = get_runtime(RuntimeContext)
    return runtime.context.db.run(query)

agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[execute_sql],
    system_prompt=SYSTEM_PROMPT,
    context_schema=RuntimeContext,
)

for step in agent.stream({"messages": question}, context=RuntimeContext(db=db), stream_mode="values"):
    step["messages"][-1].pretty_print()
```

**Takeaway:** Agents self-discover schema, self-correct errors, but don't remember between invocations.

---

## L2: Messages

**Goal:** Understand message types as the fundamental unit of LLM context.

**Message Types:**
| Type | Role | Purpose |
|------|------|---------|
| `HumanMessage` | user | User input |
| `AIMessage` | assistant | Model response |
| `ToolMessage` | tool | Tool execution results |
| `SystemMessage` | system | Behavior instructions |

**Input Formats (all valid):**
```python
# Explicit
HumanMessage("Hello")
# String (inferred)
{"messages": "Hello"}
# Dict
{"messages": {"role": "user", "content": "Hello"}}
```

**Rich Metadata Available:**
- `msg.content` - text content
- `msg.type` - message type
- `msg.usage_metadata` - token counts
- `msg.response_metadata` - model info, finish reason

**Takeaway:** Messages carry both content and metadata; `.pretty_print()` shows conversation flow.

---

## L3: Streaming

**Goal:** Reduce latency by delivering data before completion.

**Stream Modes:**

| Mode | Output | Use Case |
|------|--------|----------|
| `invoke` | Full result | No streaming |
| `stream_mode="values"` | Complete state per step | See agent reasoning |
| `stream_mode="messages"` | Token-by-token | Chatbot UX |
| `stream_mode="custom"` | User-defined data | Progress updates |

**Custom Streaming from Tools:**
```python
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    writer = get_stream_writer()
    writer(f"Looking up {city}...")  # Custom stream event
    return f"Sunny in {city}"
```

**Takeaway:** Use `messages` for lowest latency; combine modes like `["values", "custom"]`.

---

## L4: Tools

**Goal:** Enable agents to act in the real world via tools.

**Tool Definition:**
```python
from langchain.tools import tool

@tool
def calculator(a: float, b: float, operation: Literal["add", "subtract"]) -> float:
    """Perform arithmetic on two numbers."""  # Docstring = LLM instruction
    ...
```

**Enhanced Descriptions:**
```python
@tool("calculator", parse_docstring=True, description="Custom description here")
def calculator(a: float, b: float) -> float:
    """Short summary.

    Args:
        a: First operand
        b: Second operand
    """
```

**Key Points:**
- Type hints + docstrings guide LLM tool selection
- Tool descriptions significantly impact when tools are called
- `parse_docstring=True` extracts Google-style arg docs

**Takeaway:** Careful descriptions help agents discover and correctly use tools.

---

## L5: Tools with MCP

**Goal:** Connect agents to external tools via Model Context Protocol.

**MCP Setup:**
```python
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient({
    "time": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@theo.foobar/mcp-time"],
    }
})

mcp_tools = await mcp_client.get_tools()
agent = create_agent(model="openai:gpt-5", tools=mcp_tools)
```

**Takeaway:** MCP standardizes tool connections; use `await agent.ainvoke()` for async MCP tools.

---

## L6: Memory

**Goal:** Persist conversation state between invocations.

**Without Memory:** Agent forgets context, re-discovers schema each time.

**With Memory:**
```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openai:gpt-5",
    tools=[execute_sql],
    checkpointer=InMemorySaver(),  # Enable persistence
)

# Use thread_id to maintain conversation
config = {"configurable": {"thread_id": "1"}}
agent.stream({"messages": q1}, config, ...)
agent.stream({"messages": q2}, config, ...)  # Remembers q1!
```

**Takeaway:** `checkpointer` + `thread_id` = persistent conversation memory.

---

## L7: Structured Output

**Goal:** Force agent responses into defined schemas.

**Supported Formats:**
- `TypedDict`
- `pydantic.BaseModel`
- `dataclasses`
- JSON schema dict

**Usage:**
```python
class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(model="openai:gpt-5-mini", response_format=ContactInfo)
result = agent.invoke({"messages": [recorded_conversation]})
result["structured_response"]  # {'name': 'John', 'email': '...', 'phone': '...'}
```

**Takeaway:** `response_format` guarantees structured output in `result["structured_response"]`.

---

## L8: Dynamic Prompts

**Goal:** Modify system prompts at runtime based on context.

**Pattern:**
```python
from langchain.agents.middleware.types import ModelRequest, dynamic_prompt

@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    if not request.runtime.context.is_employee:
        return RESTRICTED_PROMPT
    return FULL_ACCESS_PROMPT

agent = create_agent(
    model="openai:gpt-5",
    middleware=[dynamic_system_prompt],  # Inject middleware
    context_schema=RuntimeContext,
)
```

**Use Case:** Role-based access control - employees see all tables, customers see limited subset.

**Takeaway:** Middleware + `@dynamic_prompt` enables context-aware prompt generation.

---

## L9: Human-in-the-Loop (HITL)

**Goal:** Require human approval before tool execution.

**Setup:**
```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model="openai:gpt-5",
    tools=[execute_sql],
    checkpointer=InMemorySaver(),  # Required for interrupts
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"execute_sql": {"allowed_decisions": ["approve", "reject"]}}
        ),
    ],
)
```

**Handling Interrupts:**
```python
result = agent.invoke({"messages": [question]}, config=config)

if "__interrupt__" in result:
    # Show pending action to user, get decision
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),  # or "reject"
        config=config,
    )
```

**Takeaway:** HITL adds safety gates; requires checkpointer; use `Command(resume=...)` to continue.

---

## Course Architecture Summary

```
L1: create_agent basics     --> Foundation
L2: Messages                --> Data format
L3: Streaming               --> UX optimization
L4-5: Tools & MCP           --> Agent capabilities
L6: Memory                  --> State persistence
L7: Structured Output       --> Output control
L8-9: Middleware            --> Dynamic behavior & safety
```

**Core Pattern Throughout:**
```python
agent = create_agent(
    model="provider:model-name",
    tools=[...],
    system_prompt="...",
    context_schema=RuntimeContext,
    checkpointer=InMemorySaver(),
    middleware=[...],
    response_format=Schema,
)
```
