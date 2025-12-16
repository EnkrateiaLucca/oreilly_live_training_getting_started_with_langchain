"""
Jira Agent - LangChain 1.0 Version
A simple agent for managing Jira issues using the new create_agent API.
"""
from jira import JIRA
import os
import sys
from langchain_core.tools import tool
from langchain.agents import create_agent  # LangChain 1.0: new unified agent API
from langchain_openai import ChatOpenAI


@tool
def list_issue_transition_options(issue_key: str, server_url: str = '') -> str:
    """Lists all the available transition options for a Jira issue given the issue key."""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]

    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    transitions = jira.transitions(issue_key)
    result = []
    for transition in transitions:
        result.append(f"{transition['name']}: {transition['id']}")
    return "\n".join(result)


@tool
def update_issue_status(issue_key: str, target_status_name: str, server_url: str = '') -> str:
    """Updates the status of a Jira issue given the issue key and the target status name."""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]

    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    transitions = jira.transitions(issue_key)
    target_transition_id = None

    for transition in transitions:
        if transition['name'].lower().strip() == target_status_name.lower():
            target_transition_id = transition['id']
            break

    if target_transition_id:
        jira.transition_issue(issue_key, target_transition_id)
        return f"Issue {issue_key} has been moved to '{target_status_name}'."
    else:
        return f"Transition to '{target_status_name}' not found."


@tool
def create_issue(summary: str, description: str, issue_type: str, project_key: str = 'ML', server_url: str = '') -> str:
    """Creates a Jira issue with summary, description, issue type and a project key."""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]

    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    issue_dict = {
        'project': {'key': project_key},
        'summary': summary,
        'description': description,
        'issuetype': {'name': issue_type},
    }
    new_issue = jira.create_issue(fields=issue_dict)
    return f'New issue created with key: {new_issue.key}'


@tool
def delete_issue(issue_key: str, server_url: str = '') -> str:
    """Deletes a Jira issue given the issue key."""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]
    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    issue = jira.issue(issue_key)
    delete_confirm = input(f"Do you want to delete issue {issue.key}? (y/n): ")
    if delete_confirm.lower() in ['y', 'yes']:
        issue.delete()
        return 'Issue deleted successfully'
    return 'Issue deletion cancelled'


@tool
def update_issue_summary(issue_key: str, summary: str, server_url: str = '') -> str:
    """Updates issue summary."""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]

    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    issue = jira.issue(issue_key)
    issue.update(summary=summary)
    return f'Issue {issue.key} summary updated successfully'


@tool
def update_issue_description(issue_key: str, description: str, server_url: str = '') -> str:
    """Updates issue description."""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]

    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    issue = jira.issue(issue_key)
    issue.update(description=description)
    return f'Issue {issue.key} description updated successfully'


@tool
def view_issue(issue_key: str, server_url: str = '') -> str:
    """Views a Jira issue given the issue key."""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]

    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    issue = jira.issue(issue_key)
    return f'Issue {issue.key}: {issue.fields.summary}\nStatus: {issue.fields.status}\nDescription: {issue.fields.description}'


# Define tools list
tools = [
    view_issue,
    create_issue,
    update_issue_summary,
    update_issue_description,
    delete_issue,
    update_issue_status,
    list_issue_transition_options,
]


def create_jira_agent():
    """Create a Jira management agent using LangChain 1.0 create_agent API."""
    # LangChain 1.0: Use create_agent instead of AgentExecutor + custom LCEL chain
    agent = create_agent(
        model="openai:gpt-4o-mini",  # String format for model specification
        tools=tools,
        prompt="You are a powerful assistant that helps users manage their issues in Jira Software. "
               "Use the available tools to view, create, update, and manage Jira issues.",
    )
    return agent


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python jira-agent.py '<your request>'")
        sys.exit(1)

    action_input = sys.argv[1]
    agent = create_jira_agent()

    # LangChain 1.0: Invoke agent directly with messages format
    result = agent.invoke({
        "messages": [{"role": "user", "content": action_input}]
    })

    # Print the final response
    print(result["messages"][-1].content)
