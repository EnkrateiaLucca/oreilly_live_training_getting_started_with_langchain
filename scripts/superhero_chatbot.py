"""
Superhero RAG Chatbot - A Streamlit app that lets you chat with a superhero database.

This app uses LangChain 1.0+ patterns with:
- ChromaDB for vector storage
- OpenAI embeddings for semantic search
- A retrieval tool for querying the superhero database
- An agent that can answer questions about superheroes

Run with: streamlit run scripts/superhero_chatbot.py
"""

import streamlit as st
import pandas as pd
from pathlib import Path

from langchain_community.document_loaders import CSVLoader
from langchain_chroma.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
# Use resolve() to get absolute path - critical for Streamlit which may run from different directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "notebooks" / "assets-resources" / "superheroes.csv"
PERSIST_DIRECTORY = PROJECT_ROOT / "notebooks" / "superhero-db"
COLLECTION_NAME = "superhero-collection"


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Setup (cached to avoid reloading on each interaction)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_vector_store():
    """Load or create the ChromaDB vector store for superhero data."""
    embeddings = OpenAIEmbeddings()

    # Check if persisted vector store exists
    if PERSIST_DIRECTORY.exists():
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(PERSIST_DIRECTORY)
        )
    else:
        # Load documents and create new vector store
        docs = CSVLoader(str(DATA_PATH)).load_and_split()
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(PERSIST_DIRECTORY)
        )

    return vector_store


@st.cache_data
def load_dataframe():
    """Load the superhero CSV as a pandas DataFrame for display."""
    return pd.read_csv(DATA_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Tool and Agent Setup
# ─────────────────────────────────────────────────────────────────────────────
def create_retrieval_tool(vector_store):
    """Create a retrieval tool that queries the vector store."""

    @tool
    def retrieve_superhero_info(query: str) -> str:
        """
        Retrieves information from the superhero database.
        Use this tool to find superheroes by name, superpower, power level, or catchphrase.
        Returns relevant superhero entries matching the query.
        """
        retrieved_docs = vector_store.similarity_search(query, k=5)
        if not retrieved_docs:
            return "No matching superheroes found."

        results = []
        for doc in retrieved_docs:
            results.append(doc.page_content)

        return "\n\n---\n\n".join(results)

    return retrieve_superhero_info


@st.cache_resource
def create_superhero_agent(_vector_store):
    """Create the LangChain agent with the retrieval tool."""
    retrieval_tool = create_retrieval_tool(_vector_store)

    agent = create_agent(
        model="openai:gpt-4o-mini",
        system_prompt="""You are a helpful assistant with access to a superhero database.
You can look up information about superheroes including their names, superpowers, power levels, and catchphrases.

When answering questions:
1. Use the retrieve_superhero_info tool to search the database
2. Provide clear, friendly responses based on the retrieved information
3. If multiple superheroes match a query, list them all
4. If no matches are found, let the user know and suggest alternative searches

Be enthusiastic about superheroes and engage with the user's curiosity!""",
        tools=[retrieval_tool]
    )

    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Superhero Database Chatbot",
        page_icon="🦸",
        layout="wide"
    )

    st.title("🦸 Superhero Database Chatbot")
    st.markdown("Ask me anything about superheroes! I can help you find heroes by their powers, names, or catchphrases.")

    # Sidebar with data visualization
    with st.sidebar:
        st.header("📊 Superhero Database")

        df = load_dataframe()

        # Display statistics
        st.metric("Total Superheroes", len(df))
        st.metric("Avg Power Level", f"{df['Power Level'].mean():.1f}")
        st.metric("Max Power Level", df['Power Level'].max())

        st.divider()

        # Filter options
        st.subheader("🔍 Filter & Explore")

        min_power = st.slider(
            "Minimum Power Level",
            min_value=int(df['Power Level'].min()),
            max_value=int(df['Power Level'].max()),
            value=int(df['Power Level'].min())
        )

        filtered_df = df[df['Power Level'] >= min_power]

        # Show filtered table
        st.dataframe(
            filtered_df.sort_values('Power Level', ascending=False),
            use_container_width=True,
            height=300
        )

        st.divider()

        # Power level distribution
        st.subheader("📈 Power Level Distribution")
        st.bar_chart(df['Power Level'].value_counts().sort_index())

    # Main chat area
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load resources
    vector_store = load_vector_store()
    agent = create_superhero_agent(vector_store)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about superheroes... (e.g., 'Who has the power of lightning?')"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Searching the superhero database..."):
                # Build message history for context
                messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]

                # Invoke the agent
                result = agent.invoke({"messages": messages})

                # Extract the assistant's response
                response = result["messages"][-1].content

                st.markdown(response)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Quick action buttons
    st.divider()
    st.subheader("💡 Try asking about:")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔥 Fire powers"):
            st.session_state.quick_query = "Which superheroes have fire-related powers?"
            st.rerun()

    with col2:
        if st.button("⚡ Electric powers"):
            st.session_state.quick_query = "Who can manipulate electricity or lightning?"
            st.rerun()

    with col3:
        if st.button("💪 Strongest heroes"):
            st.session_state.quick_query = "Who are the most powerful superheroes with the highest power levels?"
            st.rerun()

    with col4:
        if st.button("👻 Invisibility"):
            st.session_state.quick_query = "Which heroes can turn invisible?"
            st.rerun()

    # Handle quick query buttons
    if "quick_query" in st.session_state:
        query = st.session_state.quick_query
        del st.session_state.quick_query

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching the superhero database..."):
                messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]

                result = agent.invoke({"messages": messages})
                response = result["messages"][-1].content
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
