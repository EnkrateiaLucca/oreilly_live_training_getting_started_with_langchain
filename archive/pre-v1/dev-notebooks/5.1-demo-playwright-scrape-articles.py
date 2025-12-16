# requires installing playwright! 
import pprint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.chains import create_extraction_chain
from langchain_openai import ChatOpenAI

def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)

def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    docs = loader.load()
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["span"])
    print("Extracting content with LLM")

    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    splits = splitter.split_documents(docs_transformed)

    # Process the first split
    extracted_content = extract(schema=schema, content=splits[0].page_content)
    pprint.pprint(extracted_content)
    return extracted_content

schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

urls = ["https://www.wsj.com"]

extracted_content = scrape_with_playwright(urls, schema=schema)

print("Extracted content:")
print(extracted_content)
