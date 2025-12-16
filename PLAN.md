# LangChain Course Redesign Plan: v0.3 → v1.0 Migration

> **Project**: O'Reilly Live Training - Getting Started with LangChain
> **Created**: 2025-12-16
> **Target**: LangChain >= 1.0.0, LangGraph >= 1.0.0, Python 3.10+

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Breaking Changes in LangChain 1.0](#breaking-changes-in-langchain-10)
4. [File-by-File Migration Plan](#file-by-file-migration-plan)
5. [New Modules to Create](#new-modules-to-create)
6. [Commit Pipeline & Branch Strategy](#commit-pipeline--branch-strategy)
7. [Testing & Validation Checklist](#testing--validation-checklist)
8. [Timeline & Milestones](#timeline--milestones)

---

## Executive Summary

This plan outlines the complete migration of the LangChain training course from **v0.3.x patterns** to **v1.0+ patterns**. The migration involves:

- **9 existing notebooks** requiring updates (3 major rewrites, 6 minor updates)
- **5 new notebooks** to cover new LangChain 1.0 concepts
- **4 scripts** requiring updates
- **1 requirements file** overhaul
- Alignment with the official **LangChain Essentials** course structure

### Key Metrics

| Category | Count | Effort |
|----------|-------|--------|
| Full Rewrites | 3 notebooks | High |
| Major Updates | 4 notebooks | Medium |
| Minor Updates | 5 notebooks | Low |
| New Content | 5 notebooks | High |
| Script Updates | 4 files | Medium |
| Archive | 6+ notebooks | Low |

---

## Current State Analysis

### Current Dependencies (requirements.txt)

```
langchain==0.3.27          → UPGRADE to >=1.0.0
langchain-core==0.3.75     → UPGRADE to >=1.0.0
langchain-community==0.3.27 → Keep (continues as-is)
langchain-openai==0.2.2    → Keep (continues as-is)
langgraph==0.2.58          → UPGRADE to >=1.0.0
langchain-chroma==0.1.4    → Keep
langchain-experimental==0.3.2 → Keep
langchain-ollama==0.2.0    → Keep
langchainhub==0.1.20       → DEPRECATE (move to langchain-classic)
langserve==0.3.0           → CHECK compatibility
```

### Current File Inventory

```
notebooks/
├── 1.0-intro-to-langchain.ipynb           # UPDATE (Medium)
├── 1.1-intro-to-runnable-interface.ipynb  # UPDATE (Minor)
├── 2.0-simple-rag-langchain-langgraph.ipynb # UPDATE (Medium)
├── 3.0-building-llm-agents-with-langchain.ipynb # FULL REWRITE
├── 3.2-langchain-agent-with-langgraph.ipynb # UPDATE (Medium)
├── 4.0-langgraph-quick-introduction.ipynb # UPDATE (Minor)
├── 4.0-structured-research-report-generation.ipynb # UPDATE (Minor)
├── 5.0-demos-research-workflows.ipynb     # UPDATE (Minor)
├── intro-rag-basics.ipynb                 # UPDATE (Minor)
├── live-*.ipynb (6 files)                 # ARCHIVE
├── dev-notebooks/ (10+ files)             # ARCHIVE
├── README.md                              # UPDATE
└── langchain-lcel-cheatsheet.md           # UPDATE

scripts/
├── langchain-app.py                       # UPDATE
├── rag_methods.py                         # UPDATE
├── langchain-structured-output-ui.py      # UPDATE
├── simple-product-info-chatbot.py         # UPDATE
└── jira-agent.py                          # UPDATE

requirements/
└── requirements.txt                       # OVERHAUL
```

---

## Breaking Changes in LangChain 1.0

### 1. Agent API Complete Overhaul

#### OLD Pattern (Current in Course)
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
result = executor.invoke({"input": "query"})
```

#### NEW Pattern (LangChain 1.0)
```python
from langchain.agents import create_agent

agent = create_agent(
    model="openai:gpt-4o-mini",  # String format: "provider:model"
    tools=[tool1, tool2],
    system_prompt="You are a helpful assistant.",
    context_schema=RuntimeContext,  # Optional dependency injection
    checkpointer=InMemorySaver(),   # Optional persistence
    middleware=[...],               # Optional middleware
    response_format=Schema,         # Optional structured output
)

result = agent.invoke({"messages": "query"}, context=RuntimeContext(...))
```

### 2. Deprecated Imports

| Old Import | Status | New Import |
|------------|--------|------------|
| `from langchain.agents import create_tool_calling_agent` | DEPRECATED | `from langchain.agents import create_agent` |
| `from langchain.agents import AgentExecutor` | DEPRECATED | Built into `create_agent` |
| `from langgraph.prebuilt import create_react_agent` | DEPRECATED | `from langchain.agents import create_agent` |
| `from langchain import hub` | LEGACY | `from langchain_classic import hub` (if needed) |
| `from langchain.chains import ...` | LEGACY | `from langchain_classic.chains import ...` |

### 3. New Required Patterns

#### RuntimeContext (Dependency Injection)
```python
from dataclasses import dataclass
from langchain.agents import get_runtime

@dataclass
class RuntimeContext:
    db: SQLDatabase
    user_id: str
    is_admin: bool = False

@tool
def query_database(sql: str) -> str:
    """Execute SQL query."""
    runtime = get_runtime(RuntimeContext)
    return runtime.context.db.run(sql)
```

#### Middleware System
```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import ModelRequest, dynamic_prompt

@dynamic_prompt
def role_based_prompt(request: ModelRequest) -> str:
    if request.runtime.context.is_admin:
        return ADMIN_PROMPT
    return USER_PROMPT
```

#### Structured Output Integration
```python
agent = create_agent(
    model="openai:gpt-4o",
    response_format=MyPydanticModel,  # Integrated, single LLM call
)
result["structured_response"]  # Direct access to structured data
```

### 4. State Management Changes

| Old | New |
|-----|-----|
| Pydantic models in agent state | TypedDict only (extending AgentState) |
| Manual message handling | `{"messages": ...}` format |
| `verbose=True` for debugging | LangSmith tracing |

### 5. Python Version Requirement

- **Old**: Python 3.9+
- **New**: Python 3.10+ (3.9 EOL October 2025)

---

## File-by-File Migration Plan

### Phase 1: Core Infrastructure

#### `requirements/requirements.txt`
**Action**: OVERHAUL
**Priority**: P0 (Must be first)

```diff
- langchain==0.3.27
+ langchain>=1.0.0
- langchain-core==0.3.75
+ langchain-core>=1.0.0
- langgraph==0.2.58
+ langgraph>=1.0.0
+ langchain-classic>=0.1.0  # For legacy imports if needed

# Keep these (no breaking changes)
langchain-community==0.3.27
langchain-openai==0.2.2
langchain-chroma==0.1.4
langchain-ollama==0.2.0
langchain-experimental==0.3.2

# Add new dependencies
+ langchain-mcp-adapters>=0.1.0  # For MCP tools module

# Deprecate
- langchainhub==0.1.20  # Move to langchain-classic if needed
```

**Commit**: `chore(deps): upgrade to langchain 1.0 and langgraph 1.0`

---

### Phase 2: Notebook Updates

#### `notebooks/1.0-intro-to-langchain.ipynb`
**Action**: UPDATE (Medium effort)
**Changes**:

1. **Update model initialization section**
   ```diff
   - from langchain.chat_models import init_chat_model
   - chat_model = init_chat_model(model="gpt-4o-mini", temperature=0)
   + from langchain.agents import create_agent
   + # Show both patterns:
   + # Pattern 1: Direct model usage (still valid for chains)
   + from langchain_openai import ChatOpenAI
   + llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
   + # Pattern 2: Agent with model string (new in 1.0)
   + agent = create_agent(model="openai:gpt-4o-mini", tools=[])
   ```

2. **Add new section: "LangChain 1.0 Architecture Overview"**
   - Explain `langchain`, `langchain-core`, `langgraph` relationship
   - Introduce `create_agent` as the new primary abstraction
   - Show model string format: `"provider:model-name"`

3. **Update LCEL section**
   - LCEL patterns (`prompt | model | parser`) remain valid
   - Add note about when to use chains vs agents

4. **Add RuntimeContext introduction**
   - Brief intro to dependency injection pattern
   - Reference to detailed coverage in Module 1 rewrite

**Commit**: `docs(notebook): update 1.0-intro for langchain 1.0 patterns`

---

#### `notebooks/1.1-intro-to-runnable-interface.ipynb`
**Action**: UPDATE (Minor effort)
**Changes**:

1. **Update imports** (if any deprecated)
2. **Verify all Runnable patterns still work**
   - `RunnableLambda` ✓
   - `RunnableSequence` ✓
   - `RunnableParallel` ✓
   - `RunnablePassthrough` ✓
3. **Add note**: "These patterns are foundational to how `create_agent` works internally"

**Commit**: `docs(notebook): verify runnable patterns for langchain 1.0`

---

#### `notebooks/2.0-simple-rag-langchain-langgraph.ipynb`
**Action**: UPDATE (Medium effort)
**Changes**:

1. **Keep StateGraph patterns** (still valid in LangGraph 1.0)
2. **Update any agent usage in generation step**
   ```diff
   - # If using AgentExecutor
   + # Use create_agent for generation with tools
   ```
3. **Update deprecated langgraph imports**
   ```diff
   - from langgraph.prebuilt import ...  # Check if used
   + from langgraph.graph import ...     # Use graph primitives
   ```
4. **Add streaming section** for RAG responses

**Commit**: `docs(notebook): update RAG notebook for langgraph 1.0`

---

#### `notebooks/3.0-building-llm-agents-with-langchain.ipynb`
**Action**: FULL REWRITE
**Priority**: P1 (Critical path)
**Estimated Cells**: ~40-50

**Current Structure (to be replaced)**:
- Tool calling with `@tool` decorator
- `create_tool_calling_agent` + `AgentExecutor`
- Pydantic structured output with `with_structured_output()`
- `bind_tools()` pattern
- Retriever tools
- LangSmith tracing

**New Structure**:

```markdown
## Cell 1: Introduction
- LangChain 1.0 agent philosophy
- ReAct loop explanation
- When to use agents vs chains

## Cell 2-5: Environment Setup
- API keys
- Import new packages

## Cell 6-10: Tool Definition
- @tool decorator (unchanged)
- Type hints and docstrings
- Multiple tools example

## Cell 11-15: Basic create_agent
- First agent with create_agent()
- Model string format
- System prompt
- invoke() and stream()

## Cell 16-20: RuntimeContext
- Dependency injection pattern
- get_runtime() inside tools
- Database connection example

## Cell 21-25: Structured Output
- response_format parameter
- Pydantic model integration
- Accessing structured_response

## Cell 26-30: Memory/Checkpointing
- InMemorySaver
- thread_id for conversations
- Multi-turn example

## Cell 31-35: Tool Selection Deep Dive
- How LLM selects tools
- Description impact
- parse_docstring=True

## Cell 36-40: Real-World Example
- SQL Agent with RuntimeContext
- Streaming output
- Error handling

## Cell 41-45: Comparison with Old Patterns
- Side-by-side: AgentExecutor vs create_agent
- Migration path
- When to use langchain-classic
```

**Commit**: `feat(notebook): rewrite 3.0 agents notebook for langchain 1.0 create_agent API`

---

#### `notebooks/3.2-langchain-agent-with-langgraph.ipynb`
**Action**: UPDATE (Medium effort)
**Changes**:

1. **Update deprecated prebuilt imports**
   ```diff
   - from langgraph.prebuilt import create_react_agent
   + # create_react_agent moved to langchain.agents
   + from langchain.agents import create_agent
   ```

2. **Keep custom StateGraph examples** (still valid)

3. **Add section**: "When to use create_agent vs custom StateGraph"
   - `create_agent`: Standard agent patterns
   - Custom `StateGraph`: Complex workflows, custom state

4. **Update checkpointer patterns if needed**

**Commit**: `docs(notebook): update langgraph agent notebook for 1.0 deprecations`

---

#### `notebooks/4.0-langgraph-quick-introduction.ipynb`
**Action**: UPDATE (Minor effort)
**Changes**:

1. **Verify all StateGraph patterns** (should be unchanged)
2. **Update any deprecated imports**
3. **Add note about relationship to create_agent**

**Commit**: `docs(notebook): verify langgraph intro for 1.0 compatibility`

---

#### `notebooks/4.0-structured-research-report-generation.ipynb`
**Action**: UPDATE (Minor effort)
**Changes**:

1. **Update structured output patterns**
   ```diff
   - llm.with_structured_output(Schema)
   + # Both patterns valid:
   + # 1. llm.with_structured_output(Schema) - for chains
   + # 2. create_agent(response_format=Schema) - for agents
   ```

**Commit**: `docs(notebook): update structured output patterns for 1.0`

---

#### `notebooks/5.0-demos-research-workflows.ipynb`
**Action**: UPDATE (Minor effort)
**Changes**:

1. **Update any agent patterns used**
2. **Verify Tavily integration still works**

**Commit**: `docs(notebook): verify research workflows for 1.0 compatibility`

---

#### `notebooks/intro-rag-basics.ipynb`
**Action**: UPDATE (Minor effort)
**Changes**:

1. **Verify document loader patterns**
2. **Update any chain/agent patterns**

**Commit**: `docs(notebook): verify RAG basics for 1.0 compatibility`

---

### Phase 3: New Notebooks

#### `notebooks/2.0-messages-deep-dive.ipynb` (NEW)
**Content**:

```markdown
## Learning Objectives
- Understand message types as the unit of LLM context
- Work with message metadata
- Use .pretty_print() for debugging

## Sections
1. Message Types Overview
   - HumanMessage, AIMessage, ToolMessage, SystemMessage

2. Input Format Flexibility
   - String input: "Hello"
   - Dict input: {"role": "user", "content": "Hello"}
   - Explicit: HumanMessage("Hello")

3. Message Metadata
   - .content - text content
   - .type - message type
   - .usage_metadata - token counts
   - .response_metadata - model info
   - .content_blocks (NEW in 1.0) - multi-modal content

4. Practical Examples
   - Token tracking for cost calculation
   - Debugging with pretty_print()
```

**Commit**: `feat(notebook): add messages deep dive module for langchain 1.0`

---

#### `notebooks/3.0-streaming-patterns.ipynb` (NEW)
**Content**:

```markdown
## Learning Objectives
- Implement low-latency streaming for real-time UX
- Use custom stream events from tools
- Combine multiple stream modes

## Sections
1. Stream Modes Comparison
   - invoke() - no streaming
   - stream_mode="values" - complete state per step
   - stream_mode="messages" - token-by-token
   - stream_mode="custom" - user-defined events

2. Token-by-Token Streaming
   - Chatbot UX pattern
   - Handling partial responses

3. Custom Streaming from Tools
   - get_stream_writer()
   - Progress updates during long operations
   - Combining with values mode

4. Production Patterns
   - FastAPI streaming endpoints
   - WebSocket integration
```

**Commit**: `feat(notebook): add streaming patterns module for langchain 1.0`

---

#### `notebooks/5.0-mcp-external-tools.ipynb` (NEW)
**Content**:

```markdown
## Learning Objectives
- Connect agents to external tools via MCP
- Configure multiple MCP servers
- Handle async MCP operations

## Sections
1. MCP Introduction
   - What is Model Context Protocol?
   - Benefits over custom integrations

2. Setup and Configuration
   - langchain_mcp_adapters installation
   - MultiServerMCPClient configuration

3. Available MCP Servers
   - Time server example
   - Filesystem server example
   - Custom MCP server

4. Async Patterns
   - await mcp_client.get_tools()
   - await agent.ainvoke()

5. Combining MCP with Custom Tools
   - Hybrid tool setup
   - Tool selection behavior
```

**Commit**: `feat(notebook): add MCP external tools module for langchain 1.0`

---

#### `notebooks/8.0-dynamic-prompts-middleware.ipynb` (NEW)
**Content**:

```markdown
## Learning Objectives
- Use middleware for fine-grained agent control
- Implement dynamic prompts based on context
- Build role-based access control

## Sections
1. Middleware Architecture
   - What runs at each step
   - Built-in vs custom middleware

2. @dynamic_prompt Decorator
   - ModelRequest object
   - Accessing runtime context
   - Returning different prompts

3. Role-Based Access Control Example
   - Admin vs user prompts
   - Table access restrictions

4. Custom Middleware
   - Logging middleware
   - Rate limiting middleware
   - Content filtering middleware

5. Chaining Multiple Middleware
   - Order of execution
   - Middleware composition
```

**Commit**: `feat(notebook): add dynamic prompts middleware module for langchain 1.0`

---

#### `notebooks/9.0-human-in-the-loop.ipynb` (NEW)
**Content**:

```markdown
## Learning Objectives
- Implement human approval gates for agent actions
- Handle interrupt and resume flows
- Build production-safe agents

## Sections
1. Why HITL Matters
   - Safety for destructive operations
   - Compliance requirements
   - User trust

2. HumanInTheLoopMiddleware Setup
   - interrupt_on configuration
   - allowed_decisions
   - Requires checkpointer

3. Handling Interrupts
   - Detecting __interrupt__ in result
   - Displaying pending action to user
   - Collecting user decision

4. Resume Flow
   - Command(resume=...) pattern
   - approve/reject/edit options

5. Production Patterns
   - Webhook integration
   - Slack/Teams approval
   - Audit logging
```

**Commit**: `feat(notebook): add human-in-the-loop module for langchain 1.0`

---

### Phase 4: Script Updates

#### `scripts/langchain-app.py`
**Action**: UPDATE
**Changes**:

1. **Update LangServe patterns** (verify compatibility)
2. **Update any deprecated chain imports**

**Commit**: `fix(scripts): update langchain-app for 1.0 compatibility`

---

#### `scripts/rag_methods.py`
**Action**: UPDATE
**Changes**:

1. **Update chain patterns**
2. **Verify Streamlit integration**
3. **Update any deprecated imports**

**Commit**: `fix(scripts): update rag_methods for 1.0 compatibility`

---

#### `scripts/langchain-structured-output-ui.py`
**Action**: UPDATE
**Changes**:

1. **Update structured output patterns**

**Commit**: `fix(scripts): update structured-output-ui for 1.0`

---

#### `scripts/jira-agent.py`
**Action**: UPDATE
**Changes**:

1. **Migrate from AgentExecutor to create_agent**

**Commit**: `fix(scripts): migrate jira-agent to create_agent`

---

### Phase 5: Archive & Cleanup

#### Files to Archive
Move to `archive/pre-v1/`:

```
notebooks/live-session-intro.ipynb
notebooks/live-notebook-rag-basics.ipynb
notebooks/live-notebook-from-llms-to-actions.ipynb
notebooks/live-session-intro-rag-basics-structured-outputs.ipynb
notebooks/live-example-liner-regression-langgraph.ipynb
notebooks/6.0-live-demo-chat-with-langchain-docs-urls.ipynb
notebooks/dev-notebooks/ (entire directory)
```

**Commit**: `chore: archive pre-1.0 live session and dev notebooks`

---

#### Documentation Updates

##### `notebooks/README.md`
**Changes**:
- Update course structure
- Add LangChain 1.0 requirements
- Update learning path

##### `notebooks/langchain-lcel-cheatsheet.md`
**Changes**:
- Add create_agent patterns
- Note deprecated patterns
- Add middleware examples

##### `course_summary.md`
**Changes**:
- Update to reflect new module structure
- Add LangChain 1.0 specific patterns

**Commit**: `docs: update documentation for langchain 1.0 course structure`

---

## Commit Pipeline & Branch Strategy

### Branch Structure

```
main
└── feature/langchain-1.0-migration
    ├── phase-1/infrastructure
    ├── phase-2/notebook-updates
    ├── phase-3/new-notebooks
    ├── phase-4/script-updates
    └── phase-5/cleanup
```

### Commit Sequence

#### Phase 1: Infrastructure (PR #1)
```bash
# Branch: feature/langchain-1.0-migration/phase-1-infrastructure
git checkout -b feature/langchain-1.0-migration/phase-1-infrastructure

# Commits:
1. chore(deps): upgrade to langchain 1.0 and langgraph 1.0
2. chore: add langchain-mcp-adapters dependency
3. chore: add langchain-classic for legacy imports

# PR Title: "chore: upgrade dependencies to LangChain 1.0"
```

#### Phase 2: Notebook Updates (PR #2)
```bash
# Branch: feature/langchain-1.0-migration/phase-2-notebook-updates
git checkout -b feature/langchain-1.0-migration/phase-2-notebook-updates

# Commits (in order):
1. docs(notebook): update 1.0-intro for langchain 1.0 patterns
2. docs(notebook): verify runnable patterns for langchain 1.0
3. docs(notebook): update RAG notebook for langgraph 1.0
4. feat(notebook): rewrite 3.0 agents notebook for langchain 1.0 create_agent API
5. docs(notebook): update langgraph agent notebook for 1.0 deprecations
6. docs(notebook): verify langgraph intro for 1.0 compatibility
7. docs(notebook): update structured output patterns for 1.0
8. docs(notebook): verify research workflows for 1.0 compatibility
9. docs(notebook): verify RAG basics for 1.0 compatibility

# PR Title: "feat: update existing notebooks for LangChain 1.0"
```

#### Phase 3: New Notebooks (PR #3)
```bash
# Branch: feature/langchain-1.0-migration/phase-3-new-notebooks
git checkout -b feature/langchain-1.0-migration/phase-3-new-notebooks

# Commits (in order):
1. feat(notebook): add messages deep dive module for langchain 1.0
2. feat(notebook): add streaming patterns module for langchain 1.0
3. feat(notebook): add MCP external tools module for langchain 1.0
4. feat(notebook): add dynamic prompts middleware module for langchain 1.0
5. feat(notebook): add human-in-the-loop module for langchain 1.0

# PR Title: "feat: add new LangChain 1.0 concept modules"
```

#### Phase 4: Script Updates (PR #4)
```bash
# Branch: feature/langchain-1.0-migration/phase-4-script-updates
git checkout -b feature/langchain-1.0-migration/phase-4-script-updates

# Commits:
1. fix(scripts): update langchain-app for 1.0 compatibility
2. fix(scripts): update rag_methods for 1.0 compatibility
3. fix(scripts): update structured-output-ui for 1.0
4. fix(scripts): migrate jira-agent to create_agent

# PR Title: "fix: update scripts for LangChain 1.0 compatibility"
```

#### Phase 5: Cleanup (PR #5)
```bash
# Branch: feature/langchain-1.0-migration/phase-5-cleanup
git checkout -b feature/langchain-1.0-migration/phase-5-cleanup

# Commits:
1. chore: archive pre-1.0 live session and dev notebooks
2. docs: update documentation for langchain 1.0 course structure
3. docs: update README with new course structure
4. chore: update .gitignore for archive folder

# PR Title: "chore: cleanup and documentation for LangChain 1.0 migration"
```

### Merge Order

```
1. PR #1 (Infrastructure) → main
2. PR #2 (Notebook Updates) → main (depends on #1)
3. PR #3 (New Notebooks) → main (depends on #1)
4. PR #4 (Script Updates) → main (depends on #1)
5. PR #5 (Cleanup) → main (depends on #2, #3, #4)
```

### Commit Message Convention

```
<type>(<scope>): <description>

Types:
- feat: New feature or content
- fix: Bug fix or correction
- docs: Documentation only
- chore: Maintenance tasks
- refactor: Code restructuring

Scopes:
- deps: Dependencies
- notebook: Jupyter notebooks
- scripts: Python scripts
- docs: Documentation files
```

---

## Testing & Validation Checklist

### Pre-Migration Tests
- [ ] Run all existing notebooks to establish baseline
- [ ] Document any existing failures
- [ ] Backup current working state

### Post-Migration Tests (Per Notebook)

#### `1.0-intro-to-langchain.ipynb`
- [ ] All cells execute without errors
- [ ] Model initialization works with new patterns
- [ ] LCEL chains produce expected output
- [ ] Token tracking displays correctly

#### `3.0-building-llm-agents-with-langchain.ipynb` (CRITICAL)
- [ ] `create_agent` initializes correctly
- [ ] Tools are called as expected
- [ ] RuntimeContext injection works
- [ ] Streaming output displays
- [ ] Structured output returns correct schema
- [ ] Memory persists across invocations

#### New Notebooks
- [ ] `2.0-messages-deep-dive.ipynb` - All message types work
- [ ] `3.0-streaming-patterns.ipynb` - All stream modes work
- [ ] `5.0-mcp-external-tools.ipynb` - MCP connection successful
- [ ] `8.0-dynamic-prompts-middleware.ipynb` - Middleware executes
- [ ] `9.0-human-in-the-loop.ipynb` - Interrupt/resume flow works

### Integration Tests
- [ ] Full course flow from Module 1 to Module 9
- [ ] Scripts execute with new dependencies
- [ ] LangSmith tracing works
- [ ] Local Ollama integration works

### Environment Tests
- [ ] Fresh install from requirements.txt succeeds
- [ ] Python 3.10 compatibility verified
- [ ] Python 3.11 compatibility verified
- [ ] Python 3.12 compatibility verified

---

## Timeline & Milestones

### Milestone 1: Infrastructure Ready
- [ ] Dependencies upgraded
- [ ] Base imports working
- [ ] No import errors in any file

### Milestone 2: Core Content Updated
- [ ] All existing notebooks updated
- [ ] 3.0 agents notebook fully rewritten
- [ ] All notebooks execute without errors

### Milestone 3: New Content Complete
- [ ] All 5 new notebooks created
- [ ] New notebooks tested
- [ ] Learning path coherent

### Milestone 4: Production Ready
- [ ] Scripts updated
- [ ] Archive complete
- [ ] Documentation updated
- [ ] Full test pass

---

## Final Course Structure

```
Module 1: Build Your First Agent Fast (1.0-intro-to-langchain.ipynb - UPDATED)
Module 2: Understanding Messages (2.0-messages-deep-dive.ipynb - NEW)
Module 3: Streaming for Real-Time UX (3.0-streaming-patterns.ipynb - NEW)
Module 4: Tools Deep Dive (3.0-building-llm-agents-with-langchain.ipynb - REWRITTEN)
Module 5: External Tools with MCP (5.0-mcp-external-tools.ipynb - NEW)
Module 6: Memory & Persistence (extracted from 4.0-langgraph-quick-introduction.ipynb)
Module 7: Structured Output (4.0-structured-research-report-generation.ipynb - UPDATED)
Module 8: Dynamic Prompts with Middleware (8.0-dynamic-prompts-middleware.ipynb - NEW)
Module 9: Human-in-the-Loop Safety (9.0-human-in-the-loop.ipynb - NEW)

Supplementary:
- LCEL Deep Dive (1.1-intro-to-runnable-interface.ipynb)
- RAG Fundamentals (2.0-simple-rag-langchain-langgraph.ipynb)
- LangGraph Custom Workflows (3.2-langchain-agent-with-langgraph.ipynb)
- Research Workflows (5.0-demos-research-workflows.ipynb)
```

---

## References

- [LangChain 1.0 Release Blog](https://blog.langchain.com/langchain-langgraph-1dot0/)
- [LangChain v1 Documentation](https://docs.langchain.com/oss/python/releases/langchain-v1)
- [LangChain Agents Documentation](https://docs.langchain.com/oss/python/langchain/agents)
- [Migration Lessons (Towards Data Science)](https://towardsdatascience.com/lessons-learnt-from-upgrading-to-langchain-1-0-in-production/)
- [LangChain 1.0 Alpha Announcement](https://blog.langchain.com/langchain-langchain-1-0-alpha-releases/)
- [LangChain 1.1 Changelog](https://changelog.langchain.com/announcements/langchain-1-1)
