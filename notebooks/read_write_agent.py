from langchain.agents import create_agent
from langchain.tools import tool


@tool
def read_file(file_path: str) -> str:
    """Reads files from path"""
    with open(file_path, "r") as f:
        return f.read()

@tool
def write_file(file_path: str, contents: str) -> str:
    """Writes files to path"""
    with open(file_path, "w") as f:
        f.write(contents)
    
    return f'File written!{file_path}'


agent = create_agent(
    model="openai:gpt-5-mini",
    tools=[write_file, read_file],
)

msg = input("What is your prompt to the agent?")

for step in agent.stream({"messages": msg},stream_mode="values",):
    step["messages"][-1].pretty_print()