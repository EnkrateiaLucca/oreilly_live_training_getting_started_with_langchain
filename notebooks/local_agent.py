#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "langchain>=1.0.0",
#   "langchain-ollama",
#   "langchain-tavily",
#   "python-dotenv"
# ]
# ///

import subprocess
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv(override=True)

# All file and bash tools are sandboxed to this directory. No path may
# resolve outside of it, and bash runs with this as its cwd.
WORKSPACE_DIR = (Path(__file__).parent / "agent_workspace").resolve()
WORKSPACE_DIR.mkdir(exist_ok=True)

# Substrings that are never allowed in a bash command, regardless of
# where they appear (covers common ways to escape the sandbox or do
# destructive/irreversible damage).
BLOCKED_PATTERNS = [
    "rm -rf",
    "sudo",
    "..",
    "curl",
    "wget",
    ">/dev",
    "mkfs",
    ":(){",  # fork bomb
    "chmod -r",
    "chown -r",
]

BASH_TIMEOUT_SECONDS = 20


def _resolve_in_workspace(path: str) -> Path:
    candidate = (WORKSPACE_DIR / path).resolve()
    if WORKSPACE_DIR not in candidate.parents and candidate != WORKSPACE_DIR:
        raise ValueError(f"Path '{path}' escapes the sandboxed workspace.")
    return candidate


@tool
def read_file(path: str) -> str:
    """Read the contents of a file inside the sandboxed workspace."""
    target = _resolve_in_workspace(path)
    if not target.is_file():
        return f"Error: '{path}' is not a file in the workspace."
    return target.read_text()


@tool
def write_file(path: str, content: str) -> str:
    """Create or overwrite a file inside the sandboxed workspace with the given content."""
    target = _resolve_in_workspace(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} chars to '{path}'."


@tool
def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace the first occurrence of old_string with new_string in a workspace file."""
    target = _resolve_in_workspace(path)
    if not target.is_file():
        return f"Error: '{path}' is not a file in the workspace."
    text = target.read_text()
    if old_string not in text:
        return f"Error: old_string not found in '{path}'."
    target.write_text(text.replace(old_string, new_string, 1))
    return f"Edited '{path}'."


@tool
def delete_file(path: str) -> str:
    """Delete a file inside the sandboxed workspace."""
    target = _resolve_in_workspace(path)
    if not target.is_file():
        return f"Error: '{path}' is not a file in the workspace."
    target.unlink()
    return f"Deleted '{path}'."


@tool
def bash(command: str) -> str:
    """Run a shell command inside the sandboxed workspace directory.

    Commands are blocked if they contain dangerous patterns (sudo, rm -rf,
    network fetches, path traversal, etc). Runs with a short timeout and
    cannot access anything outside the workspace directory.
    """
    lowered = command.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lowered:
            return f"Error: command blocked (matched restricted pattern '{pattern}')."

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=WORKSPACE_DIR,
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {BASH_TIMEOUT_SECONDS}s."

    output = f"exit_code={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return output

SYS_MSG = """
You are a personal assistant and research agent.
Everytime you finish a task or request for the user you write to a file
named memory.md with a single bullet summarizing what was done in that session.
"""


search = TavilySearch(max_results=5)
llm = init_chat_model("ollama:gemma4")
tools = [search, read_file, write_file, edit_file, delete_file, bash]
agent = create_agent(model=llm, tools=tools)

USER_INPUT = "research memory in langchain and write a 3 sentences summary to a file named memory-langchain.md"

result = agent.invoke(
    {
        "messages": [
            {"role": "system",
             "content": SYS_MSG},
            {
                "role": "user",
                "content": 
                USER_INPUT
                ,
            }
        ]
    }
)
print(result["messages"][-1].content)
