"""Chat-over-docs RAG agent — deployable via `langgraph dev`."""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------------------
# Document Loading & Indexing (lazy — built on first tool call)
# ---------------------------------------------------------------------------
URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]

_retriever = None


def get_retriever():
    """Lazily build the vector store and retriever on first use."""
    global _retriever
    if _retriever is None:
        loader = WebBaseLoader(URLS)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            collection_name="course_docs",
        )
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return _retriever


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_documents(query: str) -> str:
    """Search the loaded documents for relevant information.

    Use this tool when the user asks questions about AI agents,
    prompt engineering, or related concepts from the articles.

    Args:
        query: The search query to find relevant document chunks.
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a knowledgeable AI assistant that helps users \
understand concepts from the loaded documents about AI agents and prompt \
engineering.

Rules:
- Always search the documents before answering questions about AI agents, \
prompt engineering, or related topics.
- Cite your sources when providing information from documents.
- If the documents don't contain relevant information, say so honestly.
- Be concise but thorough."""

agent = create_agent(
    model=init_chat_model("openai:gpt-5.6-terra"),
    tools=[search_documents],
    system_prompt=SYSTEM_PROMPT,
    # No checkpointer needed — langgraph dev provides persistence automatically
)
