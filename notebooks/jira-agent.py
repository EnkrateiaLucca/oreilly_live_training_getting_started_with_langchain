from jira import JIRA
import argparse
import os
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
import sys


@tool
def list_issue_transition_options(issue_key,server_url=''):
    """Lists all the available transition options for a Jira issue given the issue key"""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]
    
    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    transitions = jira.transitions(issue_key)
    for transition in transitions:
        print(transition['name'], transition['id'])

@tool
def update_issue_status(issue_key, target_status_name, server_url=''):
    """Updates the status of a Jira issue given the issue key and the target status name"""
    
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]
    
    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    transitions = jira.transitions(issue_key)
    target_transition_id = None

    # Find the transition ID for the target status
    for transition in transitions:
        print(transition['name'].lower().strip())
        if transition['name'].lower().strip() == target_status_name.lower():
            target_transition_id = transition['id']
            break

    # Execute the transition if possible
    if target_transition_id:
        jira.transition_issue(issue_key, target_transition_id)
        print(f"Issue {issue_key} has been moved to '{target_status_name}'.")
    else:
        print(f"Transition to '{target_status_name}' not found.")

@tool
def create_issue(summary, description, issue_type, project_key='ML', server_url=''):
    """Creates a Jira issue with summary, description, issue type and a project key"""
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
    print(f'New issue created with key: {new_issue.key}')


@tool
def delete_issue(issue_key, server_url=''):
    """Deletes a Jira issue given the issue key"""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]
    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    issue = jira.issue(issue_key)
    print(f'Deleting issue: {issue.key}')
    delete_issue = input("Do you want to delete the issue? (y/n): ")
    if delete_issue.lower() in ['y', 'yes']:
        issue.delete()
        print('Issue deleted successfully')


@tool
def update_issue_summary(issue_key, summary, server_url=''):
    """Updates issue summary"""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]
    
    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    issue = jira.issue(issue_key)
    issue.update(summary=summary)
    print(f'Issue {issue.key} summary updated successfully')


@tool
def update_issue_description(issue_key, description, server_url=''):
    """Updates issue description"""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]
    
    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    issue = jira.issue(issue_key)
    issue.update(description=description)
    print(f'Issue {issue.key} description updated successfully')

@tool
def view_issue(issue_key, server_url=''):
    """Views a Jira issue given the issue key"""
    JIRA_USERNAME = os.environ["JIRA_USERNAME"]
    JIRA_TOKEN = os.environ["JIRA_TOKEN"]
    
    jira = JIRA(basic_auth=(JIRA_USERNAME, JIRA_TOKEN), server=server_url)
    """Views a Jira issue given the issue key"""
    issue = jira.issue(issue_key)
    print(f'Viewing issue {issue.key}.')
    

def setup_prompt_template():
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant that helps users to manage their issues in the Jira Software.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    return prompt


def setup_agent(prompt, llm_with_tools):
    agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser())
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

if __name__=="__main__":
    action_input = sys.argv[1]
    prompt = setup_prompt_template()
    llm = ChatOpenAI()
    tools = [
        view_issue,
        create_issue,
        update_issue_summary,
        update_issue_description,
        delete_issue,
        update_issue_status,
        list_issue_transition_options,
    ]
    llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])
    agent_executor = setup_agent(prompt, llm_with_tools)
    agent_executor.invoke({"input": action_input})
