# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-community",
#     "langchain-openai",
#     "langchain-text-splitters",
#     "pypdf",
#     "python-dotenv",
# ]
# ///
"""Pre-build the document index for deployment.

Loads the PDFs in docs/, splits them, embeds each chunk, and writes
api/index_data.json so the deployed function never has to parse PDFs
or talk to a vector database — it just loads the JSON.

Run whenever the PDFs change:  uv run scripts/build_index.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env", override=True)

docs = []
for pdf_path in sorted((ROOT / "docs").glob("*.pdf")):
    docs.extend(PyPDFLoader(str(pdf_path)).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectors = embeddings.embed_documents([d.page_content for d in splits])

data = [
    {
        "text": d.page_content,
        "source": Path(d.metadata.get("source", "unknown")).name,
        "page": d.metadata.get("page", "?"),
        "embedding": v,
    }
    for d, v in zip(splits, vectors)
]

out = ROOT / "api" / "index_data.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(data))
print(f"Wrote {len(data)} chunks to {out}")
