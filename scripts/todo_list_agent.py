# /// script
# requires-python = ">=3.12"
# dependencies = ["langchain>=1.0.0", "langchain-openai>=0.3.0"]
# ///
"""
Todo List Agent - A simple task manager using LangChain 1.0+ agents.

Run with: uv run scripts/todo_list_agent.py
"""

from langchain_core.tools import tool
from langchain.agents import create_agent

# In-memory todo storage (could be replaced with a database)
TODO_LIST: dict[int, dict] = {}
_next_id = 1


@tool
def add_todo(task: str, priority: str = "medium") -> str:
    """
    Add a new task to the todo list.

    Args:
        task: Description of the task to add
        priority: Priority level - 'low', 'medium', or 'high'

    Returns:
        Confirmation message with the task ID
    """
    global _next_id
    task_id = _next_id
    TODO_LIST[task_id] = {
        "task": task,
        "priority": priority,
        "done": False
    }
    _next_id += 1
    return f"✅ Added task #{task_id}: '{task}' (priority: {priority})"


@tool
def read_todo_entries() -> str:
    """
    Read all tasks from the todo list.

    Returns:
        A formatted list of all tasks with their status
    """
    if not TODO_LIST:
        return "📋 Your todo list is empty!"

    lines = ["📋 Your Todo List:", "─" * 40]
    for task_id, entry in TODO_LIST.items():
        status = "✓" if entry["done"] else "○"
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(entry["priority"], "⚪")
        lines.append(f"  [{status}] #{task_id} {priority_emoji} {entry['task']}")

    pending = sum(1 for e in TODO_LIST.values() if not e["done"])
    lines.append("─" * 40)
    lines.append(f"Total: {len(TODO_LIST)} tasks | Pending: {pending}")
    return "\n".join(lines)


@tool
def check_todo(task_id: int) -> str:
    """
    Mark a task as completed.

    Args:
        task_id: The ID number of the task to mark as done

    Returns:
        Confirmation message or error if task not found
    """
    if task_id not in TODO_LIST:
        return f"❌ Task #{task_id} not found"

    if TODO_LIST[task_id]["done"]:
        return f"ℹ️ Task #{task_id} is already completed"

    TODO_LIST[task_id]["done"] = True
    return f"✅ Marked task #{task_id} as completed: '{TODO_LIST[task_id]['task']}'"


@tool
def delete_todo(task_id: int) -> str:
    """
    Delete a task from the todo list.

    Args:
        task_id: The ID number of the task to delete

    Returns:
        Confirmation message or error if task not found
    """
    if task_id not in TODO_LIST:
        return f"❌ Task #{task_id} not found"

    task = TODO_LIST.pop(task_id)
    return f"🗑️ Deleted task #{task_id}: '{task['task']}'"


@tool
def notify(message: str) -> str:
    """
    Send a notification/reminder about a task.

    Args:
        message: The notification message to display

    Returns:
        Confirmation that the notification was sent
    """
    print(f"\n🔔 NOTIFICATION: {message}\n")
    return f"🔔 Notification sent: '{message}'"


# Define tools list
tools = [add_todo, read_todo_entries, check_todo, delete_todo, notify]

# Create the agent using LangChain 1.0 pattern
todo_manager_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=tools,
    system_prompt="""You are a helpful task manager assistant. You help users manage their todo list.

You can:
- Add new tasks with priorities (low, medium, high)
- Show all tasks in the todo list
- Mark tasks as completed
- Delete tasks
- Send notifications/reminders

Be friendly and concise. When showing the todo list, always use the read_todo_entries tool.
When the user mentions completing or finishing a task, use check_todo.
Confirm actions clearly to the user."""
)


def chat_loop():
    """Interactive chat loop for the todo agent."""
    print("🗒️  Todo List Agent")
    print("=" * 40)
    print("Type your requests (e.g., 'add buy groceries', 'show my tasks', 'mark #1 done')")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break

        # Invoke the agent with LangChain 1.0 message format
        response = todo_manager_agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        })

        # Extract the final AI message
        final_message = response["messages"][-1]
        print(f"Agent: {final_message.content}\n")


if __name__ == "__main__":
    chat_loop()
