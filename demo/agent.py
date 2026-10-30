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
from langchain_tavily import TavilySearch

load_dotenv()

# ---------------------------------------------------------------------------
# Document Loading & Indexing (built eagerly at import time)
# ---------------------------------------------------------------------------
URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]


def _build_retriever():
    loader = WebBaseLoader(URLS)
    docs = loader.load()

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
    """Search the loaded documents for relevant information.

    Use this tool when the user asks questions about AI agents,
    prompt engineering, or related concepts from the articles.

    Args:
        query: The search query to find relevant document chunks.
    """
    docs = _retriever.invoke(query)
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

search = TavilySearch(max_results=5)
# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a knowledgeable AI assistant that helps users \
understand concepts from the loaded documents or from web search.

Rules:
- Always search the documents before answering questions about AI agents, \
prompt engineering.
- If the questions are not about agents or prompt engineering, use the web search tool
to search the web and then provide a concise response with cited sources.
- Cite your sources when providing information from documents.
- Be concise but thorough."""

agent = create_agent(
    # OpenAI's Responses API (recommended for new projects): reasoning models
    # like gpt-6-luna call function tools there without extra flags.
    model=init_chat_model("openai:gpt-6-luna", use_responses_api=True),
    tools=[search_documents, search],
    system_prompt=SYSTEM_PROMPT,
    # No checkpointer needed — langgraph dev provides persistence automatically
)
