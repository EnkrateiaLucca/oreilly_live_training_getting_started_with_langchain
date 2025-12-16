"""
Timesheet Tracking Agent using LangChain 1.0+

This agent helps users query and manage timesheet data through natural language.
"""

import csv
from pathlib import Path
from typing import Optional

from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

# Path to timesheet data
TIMESHEET_CSV = Path(__file__).parent.parent / "data" / "timesheet_data.csv"


def load_timesheet_data() -> list[dict]:
    """Load all timesheet entries from CSV."""
    with open(TIMESHEET_CSV, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_timesheet_data(entries: list[dict]) -> None:
    """Save timesheet entries back to CSV."""
    if not entries:
        return
    fieldnames = entries[0].keys()
    with open(TIMESHEET_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)


@tool
def get_all_timesheet_entries() -> str:
    """Get all timesheet entries. Returns a summary of all recorded time entries."""
    entries = load_timesheet_data()
    if not entries:
        return "No timesheet entries found."

    result = f"Found {len(entries)} timesheet entries:\n\n"
    for entry in entries[:20]:  # Limit to first 20 for readability
        result += (
            f"- ID {entry['id']}: {entry['employee_name']} | {entry['date']} | "
            f"{entry['project']} | {entry['hours_worked']}h | "
            f"{'Billable' if entry['billable'] == 'true' else 'Non-billable'}\n"
        )
    if len(entries) > 20:
        result += f"\n... and {len(entries) - 20} more entries."
    return result


@tool
def search_timesheet_entries(
    employee_name: Optional[str] = None,
    project: Optional[str] = None,
    client: Optional[str] = None,
    date: Optional[str] = None,
    billable_only: bool = False,
) -> str:
    """
    Search timesheet entries with filters.

    Args:
        employee_name: Filter by employee name (partial match)
        project: Filter by project name (partial match)
        client: Filter by client name (partial match)
        date: Filter by specific date (YYYY-MM-DD format)
        billable_only: If True, only return billable entries

    Returns:
        Matching timesheet entries with summary statistics
    """
    entries = load_timesheet_data()
    filtered = entries

    if employee_name:
        filtered = [e for e in filtered if employee_name.lower() in e["employee_name"].lower()]
    if project:
        filtered = [e for e in filtered if project.lower() in e["project"].lower()]
    if client:
        filtered = [e for e in filtered if client.lower() in e["client"].lower()]
    if date:
        filtered = [e for e in filtered if e["date"] == date]
    if billable_only:
        filtered = [e for e in filtered if e["billable"] == "true"]

    if not filtered:
        return "No matching timesheet entries found."

    # Calculate summary stats
    total_hours = sum(float(e["hours_worked"]) for e in filtered)
    billable_hours = sum(float(e["hours_worked"]) for e in filtered if e["billable"] == "true")
    total_revenue = sum(
        float(e["hours_worked"]) * float(e["hourly_rate"])
        for e in filtered if e["billable"] == "true"
    )

    result = f"Found {len(filtered)} matching entries:\n\n"
    for entry in filtered[:15]:
        result += (
            f"- ID {entry['id']}: {entry['employee_name']} | {entry['date']} | "
            f"{entry['project']} | {entry['task_description'][:40]}... | "
            f"{entry['hours_worked']}h\n"
        )
    if len(filtered) > 15:
        result += f"\n... and {len(filtered) - 15} more entries.\n"

    result += f"\n--- Summary ---\n"
    result += f"Total hours: {total_hours:.1f}h\n"
    result += f"Billable hours: {billable_hours:.1f}h\n"
    result += f"Total billable revenue: ${total_revenue:,.2f}\n"

    return result


@tool
def get_employee_summary(employee_name: str) -> str:
    """
    Get a detailed summary for a specific employee.

    Args:
        employee_name: The employee's name to look up

    Returns:
        Summary including total hours, projects worked on, and earnings
    """
    entries = load_timesheet_data()
    employee_entries = [e for e in entries if employee_name.lower() in e["employee_name"].lower()]

    if not employee_entries:
        return f"No entries found for employee matching '{employee_name}'."

    actual_name = employee_entries[0]["employee_name"]
    total_hours = sum(float(e["hours_worked"]) for e in employee_entries)
    billable_hours = sum(float(e["hours_worked"]) for e in employee_entries if e["billable"] == "true")
    total_revenue = sum(
        float(e["hours_worked"]) * float(e["hourly_rate"])
        for e in employee_entries if e["billable"] == "true"
    )

    # Get unique projects and clients
    projects = set(e["project"] for e in employee_entries)
    clients = set(e["client"] for e in employee_entries if e["client"] != "Internal")

    result = f"=== Employee Summary: {actual_name} ===\n\n"
    result += f"Total entries: {len(employee_entries)}\n"
    result += f"Total hours: {total_hours:.1f}h\n"
    result += f"Billable hours: {billable_hours:.1f}h ({billable_hours/total_hours*100:.1f}%)\n"
    result += f"Non-billable hours: {total_hours - billable_hours:.1f}h\n"
    result += f"Total billable revenue: ${total_revenue:,.2f}\n\n"
    result += f"Projects worked on ({len(projects)}): {', '.join(sorted(projects))}\n"
    result += f"Clients: {', '.join(sorted(clients)) if clients else 'None'}\n"

    return result


@tool
def add_timesheet_entry(
    employee_name: str,
    date: str,
    project: str,
    task_description: str,
    hours_worked: float,
    billable: bool,
    hourly_rate: float,
    client: str,
) -> str:
    """
    Add a new timesheet entry.

    Args:
        employee_name: Name of the employee
        date: Date of work (YYYY-MM-DD format)
        project: Project name
        task_description: Description of work done
        hours_worked: Number of hours worked (decimal)
        billable: Whether the time is billable
        hourly_rate: Hourly rate (0 for non-billable)
        client: Client name (use "Internal" for non-billable)

    Returns:
        Confirmation message with the new entry details
    """
    entries = load_timesheet_data()

    # Generate new ID
    max_id = max(int(e["id"]) for e in entries) if entries else 0
    new_id = max_id + 1

    new_entry = {
        "id": str(new_id),
        "employee_name": employee_name,
        "date": date,
        "project": project,
        "task_description": task_description,
        "hours_worked": str(hours_worked),
        "billable": "true" if billable else "false",
        "hourly_rate": str(hourly_rate),
        "client": client,
    }

    entries.append(new_entry)
    save_timesheet_data(entries)

    return (
        f"Successfully added timesheet entry (ID: {new_id}):\n"
        f"- Employee: {employee_name}\n"
        f"- Date: {date}\n"
        f"- Project: {project}\n"
        f"- Task: {task_description}\n"
        f"- Hours: {hours_worked}h\n"
        f"- Billable: {'Yes' if billable else 'No'}\n"
        f"- Rate: ${hourly_rate}/hr\n"
        f"- Client: {client}"
    )


@tool
def update_timesheet_entry(
    entry_id: int,
    hours_worked: Optional[float] = None,
    task_description: Optional[str] = None,
    billable: Optional[bool] = None,
) -> str:
    """
    Update an existing timesheet entry.

    Args:
        entry_id: The ID of the entry to update
        hours_worked: New hours worked value (optional)
        task_description: New task description (optional)
        billable: New billable status (optional)

    Returns:
        Confirmation of the update or error message
    """
    entries = load_timesheet_data()

    for entry in entries:
        if int(entry["id"]) == entry_id:
            if hours_worked is not None:
                entry["hours_worked"] = str(hours_worked)
            if task_description is not None:
                entry["task_description"] = task_description
            if billable is not None:
                entry["billable"] = "true" if billable else "false"
                if not billable:
                    entry["hourly_rate"] = "0"
                    entry["client"] = "Internal"

            save_timesheet_data(entries)
            return f"Successfully updated entry ID {entry_id}."

    return f"Entry with ID {entry_id} not found."


@tool
def get_project_summary(project_name: str) -> str:
    """
    Get a summary of hours and revenue for a specific project.

    Args:
        project_name: The project name to look up (partial match supported)

    Returns:
        Project summary with hours by employee and total revenue
    """
    entries = load_timesheet_data()
    project_entries = [e for e in entries if project_name.lower() in e["project"].lower()]

    if not project_entries:
        return f"No entries found for project matching '{project_name}'."

    actual_project = project_entries[0]["project"]
    client = project_entries[0]["client"]

    # Calculate by employee
    employee_hours = {}
    for e in project_entries:
        name = e["employee_name"]
        hours = float(e["hours_worked"])
        employee_hours[name] = employee_hours.get(name, 0) + hours

    total_hours = sum(float(e["hours_worked"]) for e in project_entries)
    total_revenue = sum(
        float(e["hours_worked"]) * float(e["hourly_rate"])
        for e in project_entries if e["billable"] == "true"
    )

    result = f"=== Project Summary: {actual_project} ===\n"
    result += f"Client: {client}\n\n"
    result += f"Hours by employee:\n"
    for name, hours in sorted(employee_hours.items(), key=lambda x: -x[1]):
        result += f"  - {name}: {hours:.1f}h\n"
    result += f"\nTotal hours: {total_hours:.1f}h\n"
    result += f"Total revenue: ${total_revenue:,.2f}\n"

    return result


# Define the tools list
timesheet_tools = [
    get_all_timesheet_entries,
    search_timesheet_entries,
    get_employee_summary,
    get_project_summary,
    add_timesheet_entry,
    update_timesheet_entry,
]

# Create memory checkpointer for conversation persistence
memory = MemorySaver()

# Create the timesheet agent using LangChain 1.0 pattern with memory
timesheet_agent = create_agent(
    model="openai:gpt-5-mini",
    tools=timesheet_tools,
    system_prompt="""You are a helpful timesheet management assistant. You help users:
- Query and search timesheet entries
- Get summaries by employee or project
- Add new time entries
- Update existing entries

When users ask about hours, revenue, or time tracking, use the appropriate tools to find the information.
Always be precise with numbers and provide clear summaries.

If a user wants to add time, make sure to gather all required information:
- Employee name, date, project, task description, hours, billable status, rate, and client.
""",
    checkpointer=memory,  # Enable conversation memory
)


def chat_loop():
    """Interactive chat loop for the timesheet agent with memory."""
    import uuid

    # Generate a unique thread ID for this conversation session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 50)
    print("Timesheet Tracking Agent (with Memory)")
    print("Ask me about timesheet entries, employee hours, or project summaries!")
    print("I'll remember our conversation context.")
    print("Type 'quit' to exit.")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Invoke the agent with LangChain 1.0 message format and thread config
        # The checkpointer will automatically persist message history for this thread
        response = timesheet_agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )

        # Extract the final response
        final_message = response["messages"][-1]
        print(f"\nAssistant: {final_message.content}")


if __name__ == "__main__":
    chat_loop()
