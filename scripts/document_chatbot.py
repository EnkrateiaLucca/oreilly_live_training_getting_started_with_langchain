# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "langchain>=1.0.0",
#     "langchain-openai>=1.0.0",
#     "langchain-core>=1.0.0",
#     "langchain-chroma>=1.0.0",
#     "pypdf>=4.0.0",
#     "pandas>=2.0.0",
#     "prompt_toolkit>=3.0.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Document Chatbot with prompt_toolkit and LangChain 1.0+

A CLI chatbot that can answer questions about PDF and CSV files using
LangChain 1.0's create_agent and prompt_toolkit for a rich terminal experience.

Usage:
    uv run document_chatbot.py

Features:
    - PDF document Q&A via semantic search (RAG)
    - CSV data analysis and querying
    - Interactive CLI with history and auto-suggestions
    - Streaming responses
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool, tool
from langchain_openai import OpenAIEmbeddings
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# --- Configuration ---
ASSETS_DIR = Path(__file__).parent.parent / "notebooks" / "assets-resources"
PDF_PATH = ASSETS_DIR / "attention-paper.pdf"
CSV_PATH = ASSETS_DIR / "superheroes.csv"
HISTORY_FILE = Path.home() / ".document_chatbot_history"

# --- Styling for prompt_toolkit ---
CHAT_STYLE = Style.from_dict({
    "prompt": "#00aa00 bold",      # Green prompt
    "user-input": "#ffffff",        # White user input
    "bot-prefix": "#0088ff bold",   # Blue bot prefix
    "error": "#ff0000 bold",        # Red errors
    "info": "#888888 italic",       # Gray info text
})


def load_pdf_to_documents(pdf_path: Path) -> list[Document]:
    """Load PDF file and convert to LangChain Documents."""
    reader = PdfReader(pdf_path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": str(pdf_path), "page": i + 1}
            ))

    return documents


def create_pdf_retriever(documents: list[Document]):
    """Create a Chroma vector store retriever from documents."""
    embeddings = OpenAIEmbeddings()

    # Create in-memory Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="pdf_documents"
    )

    return vectorstore.as_retriever(search_kwargs={"k": 4})


def load_csv_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV file into a pandas DataFrame."""
    return pd.read_csv(csv_path)


# --- Global state for tools ---
_csv_df: pd.DataFrame | None = None


@tool
def query_csv(query: str) -> str:
    """
    Query the superheroes CSV data. Use this tool when the user asks about
    superheroes, their powers, power levels, or catchphrases.

    The CSV contains columns: Superhero Name, Superpower, Power Level, Catchphrase

    Args:
        query: A description of what information to find (e.g., "strongest superhero",
               "heroes with fire powers", "superhero catchphrases")

    Returns:
        A string containing the relevant data from the CSV
    """
    global _csv_df

    if _csv_df is None:
        return "Error: CSV data not loaded"

    query_lower = query.lower()
    df = _csv_df

    # Handle common query patterns
    if "strongest" in query_lower or "highest power" in query_lower:
        top = df.nlargest(5, "Power Level")
        return f"Top 5 strongest superheroes:\n{top.to_string(index=False)}"

    elif "weakest" in query_lower or "lowest power" in query_lower:
        bottom = df.nsmallest(5, "Power Level")
        return f"Bottom 5 superheroes by power level:\n{bottom.to_string(index=False)}"

    elif "fire" in query_lower or "pyro" in query_lower:
        fire_heroes = df[df["Superpower"].str.lower().str.contains("fire|pyro", na=False)]
        if fire_heroes.empty:
            return "No fire-related superheroes found."
        return f"Fire-related superheroes:\n{fire_heroes.to_string(index=False)}"

    elif "ice" in query_lower or "frost" in query_lower or "cryo" in query_lower:
        ice_heroes = df[df["Superpower"].str.lower().str.contains("ice|frost|cryo", na=False)]
        if ice_heroes.empty:
            return "No ice-related superheroes found."
        return f"Ice-related superheroes:\n{ice_heroes.to_string(index=False)}"

    elif "all" in query_lower and "hero" in query_lower:
        return f"All superheroes:\n{df.to_string(index=False)}"

    elif "count" in query_lower or "how many" in query_lower:
        return f"There are {len(df)} superheroes in the database."

    elif "column" in query_lower or "fields" in query_lower:
        return f"CSV columns: {', '.join(df.columns.tolist())}"

    elif "average" in query_lower and "power" in query_lower:
        avg = df["Power Level"].mean()
        return f"Average power level: {avg:.1f}"

    else:
        # Generic search across all text columns
        mask = df.apply(
            lambda row: query_lower in str(row.values).lower(),
            axis=1
        )
        matches = df[mask]

        if matches.empty:
            # Return sample data to help the LLM
            sample = df.sample(min(3, len(df)))
            return f"No exact matches for '{query}'. Here's a sample of the data:\n{sample.to_string(index=False)}"

        return f"Search results for '{query}':\n{matches.to_string(index=False)}"


@tool
def csv_statistics() -> str:
    """
    Get statistical summary of the superheroes CSV data.
    Use this when the user asks for overall statistics or summaries about the data.

    Returns:
        Statistical summary including count, mean, min, max power levels
    """
    global _csv_df

    if _csv_df is None:
        return "Error: CSV data not loaded"

    stats = _csv_df["Power Level"].describe()
    return f"""Superhero Data Statistics:
- Total heroes: {int(stats['count'])}
- Average power level: {stats['mean']:.1f}
- Min power level: {int(stats['min'])}
- Max power level: {int(stats['max'])}
- Standard deviation: {stats['std']:.1f}"""


def create_agent_with_tools(pdf_retriever):
    """Create a LangChain 1.0 agent with PDF and CSV tools."""

    # Create PDF retriever tool
    pdf_tool = create_retriever_tool(
        pdf_retriever,
        name="search_pdf_document",
        description="""Search the PDF document (Attention Is All You Need paper)
        for information. Use this when the user asks about transformers, attention
        mechanisms, neural network architecture, or anything from the research paper.""",
    )

    tools = [pdf_tool, query_csv, csv_statistics]

    # Create agent using LangChain 1.0 pattern
    agent = create_agent(
        model="openai:gpt-4o-mini",
        tools=tools,
        system_prompt="""You are a helpful document assistant that can answer questions about:

1. A PDF document: "Attention Is All You Need" - the seminal transformer paper
2. A CSV file: Superhero database with names, powers, power levels, and catchphrases

Use the appropriate tools to find information:
- Use 'search_pdf_document' for questions about transformers, attention, neural networks
- Use 'query_csv' for questions about superheroes, their powers, or attributes
- Use 'csv_statistics' for statistical summaries of the superhero data

Be concise and helpful. If you can't find information, say so honestly.""",
    )

    return agent


def print_styled(text: str, style: str = ""):
    """Print text with optional ANSI styling."""
    colors = {
        "green": "\033[92m",
        "blue": "\033[94m",
        "red": "\033[91m",
        "gray": "\033[90m",
        "bold": "\033[1m",
        "reset": "\033[0m",
    }

    if style:
        print(f"{colors.get(style, '')}{text}{colors['reset']}")
    else:
        print(text)


def stream_agent_response(agent, user_message: str, messages: list):
    """Stream the agent's response and collect the full response."""
    messages.append({"role": "user", "content": user_message})

    print_styled("\n🤖 Assistant: ", "blue")

    full_response = ""

    try:
        # Use streaming for better UX
        for chunk in agent.stream({"messages": messages}):
            if "messages" in chunk:
                for msg in chunk["messages"]:
                    if hasattr(msg, 'content') and msg.content:
                        print(msg.content, end="", flush=True)
                        full_response = msg.content

        print()  # Newline after response

        if full_response:
            messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        print_styled(f"\nError: {e}", "red")

    return messages


def main():
    """Main entry point for the document chatbot."""
    global _csv_df

    print_styled("\n" + "=" * 60, "blue")
    print_styled("  📚 Document Chatbot (LangChain 1.0 + prompt_toolkit)", "bold")
    print_styled("=" * 60, "blue")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print_styled("\n❌ Error: OPENAI_API_KEY not set in environment", "red")
        print_styled("Set it with: export OPENAI_API_KEY='your-key'", "gray")
        return

    # Load documents
    print_styled("\n📄 Loading documents...", "gray")

    try:
        pdf_docs = load_pdf_to_documents(PDF_PATH)
        print_styled(f"   ✓ Loaded PDF: {PDF_PATH.name} ({len(pdf_docs)} pages)", "green")
    except FileNotFoundError:
        print_styled(f"   ⚠ PDF not found: {PDF_PATH}", "red")
        pdf_docs = [Document(page_content="No PDF document loaded.", metadata={})]

    try:
        _csv_df = load_csv_data(CSV_PATH)
        print_styled(f"   ✓ Loaded CSV: {CSV_PATH.name} ({len(_csv_df)} rows)", "green")
    except FileNotFoundError:
        print_styled(f"   ⚠ CSV not found: {CSV_PATH}", "red")

    # Create retriever and agent
    print_styled("🔧 Creating vector store and agent...", "gray")
    pdf_retriever = create_pdf_retriever(pdf_docs)
    agent = create_agent_with_tools(pdf_retriever)
    print_styled("   ✓ Agent ready!", "green")

    # Set up prompt session with history
    session = PromptSession(
        history=FileHistory(str(HISTORY_FILE)),
        auto_suggest=AutoSuggestFromHistory(),
        style=CHAT_STYLE,
    )

    # Print help
    print_styled("\n" + "-" * 60, "gray")
    print_styled("Commands:", "bold")
    print_styled("  Type your question and press Enter", "gray")
    print_styled("  Type 'quit' or 'exit' to leave", "gray")
    print_styled("  Use ↑/↓ arrows to navigate history", "gray")
    print_styled("-" * 60 + "\n", "gray")

    # Conversation history
    messages = []

    # Chat loop
    while True:
        try:
            user_input = session.prompt(
                [("class:prompt", "You: ")],
            ).strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                print_styled("\n👋 Goodbye!", "green")
                break

            messages = stream_agent_response(agent, user_input, messages)

        except KeyboardInterrupt:
            print_styled("\n\n👋 Goodbye!", "green")
            break
        except EOFError:
            print_styled("\n\n👋 Goodbye!", "green")
            break


if __name__ == "__main__":
    main()
