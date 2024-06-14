# in rag-pdf-app/app/chain.py

# inspired by this template from langchain: https://github.com/langchain-ai/langchain/blob/master/templates/rag-chroma-private/rag_chroma_private/chain.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from typing import List, Tuple

def load_pdf(file_path: str="./paper.pdf") -> str:
    loader = PyPDFLoader(file_path)
    return loader.load()

def index_docs(docs: List[str], 
                persist_directory: str="./i-shall-persist", 
                embedding_model: str="llama3"):
    embeddings = OllamaEmbeddings(model=embedding_model)
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    retriever = vectordb.as_retriever()
    
    return retriever


file_path = "./paper.pdf"

docs = load_pdf(file_path)

retriever = index_docs(docs)

template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOllama(model="llama3")

# 2 suggestions for creating the rag chain:

# chain = (
#     RunnableParallel({"context": retriever, "question": RunnablePassthrough()}) # RunnablePassthrough source: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html#langchain-core-runnables-passthrough-runnablepassthrough:~:text=Runnable%20to%20passthrough,and%20experiment%20with.
#     # RunnableParallel source: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.RunnableParallel.html
#     | prompt
#     | llm
#     | StrOutputParser()
# )

chain = RetrievalQA.from_chain_type(llm, 
                                    retriever=retriever, 
                                    chain_type_kwargs={"prompt": prompt}, 
                                    return_source_documents=True)
# qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)

# Add typing for input
class Question(BaseModel):
    __root__: str
    # The __root__ field in Pydantic models is used to define a model
    # where you expect a single value or a list rather than a dictionary 
    # of named fields. Essentially, it allows your model to handle instances 
    # where data does not naturally fit into a key-value structure, 
    # such as a single value or a list.


rag_chain = chain.with_types(input_type=Question)