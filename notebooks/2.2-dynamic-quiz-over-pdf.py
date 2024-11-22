#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('pip', 'install langchain')
get_ipython().run_line_magic('pip', 'install langchain-openai')
get_ipython().run_line_magic('pip', 'install langchainhub')
get_ipython().run_line_magic('pip', 'install pypdf')
get_ipython().run_line_magic('pip', 'install chromadb')


# In[ ]:


import os

# Set OPENAI API Key

os.environ["OPENAI_API_KEY"] = "your openai key"

# OR (load from .env file)

# from dotenv import load_dotenv
# make sure you have python-dotenv installed
# load_dotenv("./.env")


# Let's set up a study workflow using Jupyter Notebooks, LLMs, and langchain.

# In[15]:


import os
from langchain.document_loaders import PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# In[6]:


MODEL="gpt-4o-mini"


# In[7]:


pdf_path = "./assets-resources/attention-paper.pdf"
loader = PyPDFLoader(pdf_path) # LOAD
pdf_docs = loader.load_and_split() # SPLIT
pdf_docs


# In[8]:


doc_obj = pdf_docs[0]
doc_obj


# In[9]:


from IPython.display import display, Markdown

Markdown(doc_obj.page_content)


# In[14]:


embeddings = OpenAIEmbeddings() # EMBED
embeddings
vectordb = Chroma.from_documents(pdf_docs, embedding=embeddings) # STORE


# Definition of a [retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/#:~:text=A%20retriever%20is,Document's%20as%20output.):

# > A retriever is an interface that returns documents given an unstructured query. It is more general than a vector store. A retriever does not need to be able to store documents, only to return (or retrieve) them. Vector stores can be used as the backbone of a retriever, but there are other types of retrievers as well.
retriever = vectordb.as_retriever() 
# retriever
llm = ChatOpenAI(model=MODEL, temperature=0)
# source: https://python.langchain.com/v0.2/docs/tutorials/pdf_qa/#question-answering-with-rag

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)


# In[16]:


# question_answer_chain
# This method `create_stuff_documents_chain` [outputs an LCEL runnable](https://arc.net/l/quote/bnsztwth)
query = "What are the key components of the transformer architecture?"
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": query})

results


# In[17]:


from IPython.display import Markdown

final_answer = results["answer"]

Markdown(final_answer)


# In[18]:


query_summary = "Write a simple bullet points summary about this paper"

 # adding chat history so the model remembers previous questions
output = rag_chain.invoke({"input": query_summary})

Markdown(output["answer"])


# In[20]:


def ask_pdf(pdf_qa,query):
    print("QUERY: ",query)
    result = pdf_qa.invoke({"input": query})
    answer = result["answer"]
    print("ANSWER", answer)
    return answer


ask_pdf(rag_chain,"How does the self-attention mechanism in transformers differ from traditional sequence alignment methods?")


# In[21]:


quiz_questions = ask_pdf(rag_chain, "Quiz me with 3 simple questions on the positional encodings and the role they play in transformers.")

quiz_questions


# In[22]:


llm = ChatOpenAI(model=MODEL, temperature=0.0)


# In[23]:


from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class QA(BaseModel):
    questions: List[str] = Field(description='List of questions about a given context.')


# In[27]:


from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

template = f"You transform unstructured questions about a topic into a structured list of questions."
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template("Return ONLY a PYTHON list containing the questions in this text: {questions}")


# In[28]:


from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])


# In[29]:


quiz_chain = chat_prompt | llm.with_structured_output(QA)


# In[32]:


questions_list = quiz_chain.invoke({"questions": quiz_questions})
questions_list


# In[34]:


questions = questions_list.questions


# In[35]:


questions


# In[36]:


# the questions variable was created within the string inside the `questions_list` variable.
answers = []
for q in questions:
    answers.append(ask_pdf(rag_chain,q))


# In[55]:


evaluations = []

for q,a in zip(questions, answers):
    # Check for results
    evaluations.append(ask_pdf(rag_chain,f"Is this: {a} the correct answer to this question: {q} according to the paper? Return ONLY '''YES''' or '''NO'''. Output:"))

evaluations


# In[56]:


scores = []

yes_count = evaluations.count('YES')
score = str(yes_count/len(evaluations) * 100) + "%"
print(score)


# A more 'langchain way' to do this would be:

# In[48]:


from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt_eval_template = """
You take in context, a question and a generated answer and you output ONLY a score of YES if the answer is correct,
or NO if the answer is not correct.

<context>
{context}
<context>

<question>
{question}
<question>

<answer>
{answer}
<answer>
"""

prompt_eval = ChatPromptTemplate.from_template(prompt_eval_template)

answer_eval_chain = (
    {
        'context': lambda x: format_docs(x['context']),
        'question': lambda x: x['question'],
        'answer': lambda x: x['answer']
        }
    ) | prompt_eval | llm | StrOutputParser()


# In[51]:


evaluations = []
for q,a in zip(questions, answers):
    evaluations.append(answer_eval_chain.invoke({'context': pdf_docs, 'question': q, 'answer': a}))


# In[52]:


evaluations


# In[54]:


scores = []

yes_count = evaluations.count('YES')
score = str(yes_count/len(evaluations) * 100) + "%"
print(score)


# In this example notebook we introduced a few interesting ideas:
# 1. Structured outputs
# 2. Some simple evaluation of rag answers using the 'llm-as-a-judge' strategy
