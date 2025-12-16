"""
LangChain Structured Output UI - LangChain 1.0 Compatible
Demonstrates using with_structured_output() for parsing raw text into Pydantic models.
"""
import streamlit as st
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import pandas as pd


if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["Product Name", "Price", "Description"])

class Product(BaseModel):
    name: str = Field(description="The name of the product")
    price: float = Field(description="The price of the product")
    description: str = Field(description="A detailed description of the product")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

llm_structured_output = llm.with_structured_output(Product)


st.title("LangChain Structured Output UI")

col1, col2 = st.columns(2)


raw_text = st.text_area("Raw Text")

# Add a checkbox to show the current table
show_table = st.checkbox("Show current table")

# Display the current table if the checkbox is checked
if show_table:
    st.write("Current Product Information")
    st.table(st.session_state.df)

if raw_text:
    st.write("Product Information")
    if raw_text!="" and st.button("Parse"):
        product_info = llm_structured_output.invoke(raw_text)
        st.write("Product Name,Price,Description")
        new_row = pd.DataFrame({
            "Product Name": [product_info.name],
            "Price": [product_info.price],
            "Description": [product_info.description]
        })
        st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
        st.table(st.session_state.df)
    else:
        st.write("Enter text in the left column to see it here.")