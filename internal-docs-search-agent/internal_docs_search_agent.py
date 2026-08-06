# Allow the agent to answer questions about the documents
"""Chat-over-docs RAG agent — deployable via `langgraph dev`."""

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_tavily import TavilySearch

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Document Loading & Indexing (built eagerly at import time)
# ---------------------------------------------------------------------------
DOCS_DIR = Path(__file__).parent / "docs"


def _build_retriever():
    docs = []
    for pdf_path in sorted(DOCS_DIR.glob("*.pdf")):
        docs.extend(PyPDFLoader(str(pdf_path)).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="course_docs",
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


_retriever = _build_retriever()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_documents(query: str) -> str:
    """Search the internal PDF documents for relevant information.

    Use this tool when the user asks questions about LangChain or
    related concepts covered in the course PDFs.

    Args:
        query: The search query to find relevant document chunks.
    """
    docs = _retriever.invoke(query)
    return "\n\n".join(
        f"[Source: {Path(d.metadata.get('source', 'unknown')).name}, "
        f"page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

search = TavilySearch(max_results=5)
# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a knowledgeable AI assistant that helps users \
understand concepts from the internal PDF documents or from web search.

Rules:
- Always search the documents before answering questions about LangChain \
or topics covered in the course materials.
- If the question is not covered by the documents, say so.
- Cite your sources (document name and page) when providing information \
from documents.
- Be concise but thorough."""

agent = create_agent(
    # Reasoning models need reasoning_effort="none" to use function tools
    # on the /v1/chat/completions endpoint.
    model=init_chat_model("openai:gpt-5.6-luna", reasoning_effort="none"),
    tools=[search_documents],
    system_prompt=SYSTEM_PROMPT,
    # No checkpointer needed — langgraph dev provides persistence automatically
)

# Interface with the agent