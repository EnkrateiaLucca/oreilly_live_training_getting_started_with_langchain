"""Vercel-deployable version of the internal docs search agent.

Same agent as internal_docs_search_agent.py, but retrieval reads the
pre-built index_data.json (see scripts/build_index.py) instead of
building a Chroma index at import time, and the whole thing is exposed
through FastAPI — Vercel auto-detects a root-level main.py FastAPI app
and routes all requests to it.
"""

import json
import re
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Retrieval over the pre-built index
# ---------------------------------------------------------------------------
_DATA = json.loads((Path(__file__).parent / "index_data.json").read_text())
_VECTORS = np.array([chunk["embedding"] for chunk in _DATA])
_NORMS = np.linalg.norm(_VECTORS, axis=1)
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


@tool
def search_documents(query: str) -> str:
    """Search the internal PDF documents for relevant information.

    Use this tool when the user asks questions about LangChain or
    related concepts covered in the course PDFs.

    Args:
        query: The search query to find relevant document chunks.
    """
    q = np.array(_embeddings.embed_query(query))
    sims = _VECTORS @ q / (_NORMS * np.linalg.norm(q))
    top = np.argsort(sims)[-4:][::-1]
    return "\n\n".join(
        f"[Source: {_DATA[i]['source']}, page {_DATA[i]['page']}]\n{_DATA[i]['text']}"
        for i in top
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a knowledgeable AI assistant that helps users \
understand concepts from the internal PDF documents.

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
)

# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
app = FastAPI()


class ChatRequest(BaseModel):
    # Full conversation so far: [{"role": "user"|"assistant", "content": "..."}]
    messages: list[dict]


# Matches the citation lines search_documents emits in its tool output.
_SOURCE_RE = re.compile(r"\[Source: (.+?), page (\d+)\]")


@app.post("/api/chat")
def chat(req: ChatRequest):
    result = agent.invoke({"messages": req.messages})
    msgs = result["messages"]

    # Collect the chunks retrieved for THIS turn only (tool messages after
    # the last human message), so the UI can link to the cited PDF pages.
    last_human = max(i for i, m in enumerate(msgs) if m.type == "human")
    sources, seen = [], set()
    for m in msgs[last_human:]:
        if m.type == "tool":
            for name, page in _SOURCE_RE.findall(str(m.content)):
                if (name, page) not in seen:
                    seen.add((name, page))
                    # PyPDF pages are 0-indexed; browser #page= is 1-indexed.
                    sources.append({"file": name, "page": int(page) + 1})
    return {"reply": msgs[-1].content, "sources": sources}


CHAT_PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Internal Docs Search Agent</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css">
<script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, sans-serif; margin: 0; height: 100vh; display: flex; flex-direction: column; background: #faf7f0; color: #1a1a1a; }
  header { padding: .7rem 1rem; font-weight: 600; border-bottom: 1px solid #ddd; background: #fff; }
  main { flex: 1; display: flex; min-height: 0; }
  #chat { flex: 1; display: flex; flex-direction: column; min-width: 0; border-right: 1px solid #ddd; }
  #log { flex: 1; overflow-y: auto; padding: 1rem; }
  .msg { margin-bottom: 1rem; }
  .msg > strong { display: block; font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; color: #8a7a55; margin-bottom: .15rem; }
  .user .body { white-space: pre-wrap; font-weight: 500; }
  .assistant .body { color: #222; line-height: 1.5; }
  .body p { margin: .35rem 0; }
  .body ul, .body ol { margin: .35rem 0; padding-left: 1.4rem; }
  .body pre { background: #0d1117; color: #e6edf3; padding: .8rem 1rem; border-radius: 8px; overflow-x: auto; font-size: .85rem; line-height: 1.45; }
  .body code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .88em; background: #efe8d6; padding: .12em .35em; border-radius: 4px; }
  .body pre code { background: none; padding: 0; font-size: 1em; }
  .body table { border-collapse: collapse; margin: .5rem 0; font-size: .9rem; }
  .body th, .body td { border: 1px solid #ccc; padding: .3rem .6rem; }
  .body blockquote { margin: .5rem 0; padding-left: .8rem; border-left: 3px solid #c9b891; color: #555; }
  .sources { margin: .3rem 0 .8rem; }
  .src { display: inline-block; margin: .15rem .3rem .15rem 0; padding: .25rem .6rem; font-size: .8rem; border: 1px solid #c9b891; border-radius: 999px; background: #f3ead6; cursor: pointer; }
  .src:hover, .src.active { background: #1a1a1a; color: #fff; border-color: #1a1a1a; }
  form { display: flex; gap: .5rem; padding: .8rem; border-top: 1px solid #ddd; background: #fff; }
  input { flex: 1; padding: .6rem; border: 1px solid #ccc; border-radius: 6px; font-size: 1rem; }
  button { padding: .6rem 1.2rem; border: 0; border-radius: 6px; background: #1a1a1a; color: #fff; cursor: pointer; }
  button:disabled { opacity: .5; }
  #viewerPane { flex: 1; display: flex; flex-direction: column; min-width: 0; background: #eee; }
  #viewerLabel { padding: .4rem .8rem; font-size: .8rem; color: #555; background: #fff; border-bottom: 1px solid #ddd; }
  #viewer { flex: 1; border: 0; width: 100%; }
  @media (max-width: 800px) {
    main { flex-direction: column; }
    #chat { border-right: 0; }
    #viewerPane { height: 45vh; flex: none; }
  }
</style>
</head>
<body>
<header>📚 Internal Docs Search Agent</header>
<main>
  <div id="chat">
    <div id="log"></div>
    <form id="f">
      <input id="q" placeholder="Ask about the course PDFs..." autocomplete="off" autofocus>
      <button id="b">Send</button>
    </form>
  </div>
  <div id="viewerPane">
    <div id="viewerLabel">Cited pages will open here — click a source chip in the chat.</div>
    <iframe id="viewer" title="PDF viewer"></iframe>
  </div>
</main>
<script>
const messages = [];
const log = document.getElementById('log');
function add(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const label = document.createElement('strong');
  label.textContent = role === 'user' ? 'You' : 'Agent';
  const body = document.createElement('div');
  body.className = 'body';
  body.textContent = text;
  div.appendChild(label);
  div.appendChild(body);
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return body;
}
function renderMarkdown(el, text) {
  if (window.marked && window.DOMPurify) {
    el.innerHTML = DOMPurify.sanitize(marked.parse(text));
    if (window.hljs) el.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b));
  } else {
    el.textContent = text;  // CDN blocked — fall back to plain text
  }
  log.scrollTop = log.scrollHeight;
}
function showSource(file, page, chip) {
  document.querySelectorAll('.src.active').forEach(el => el.classList.remove('active'));
  if (chip) chip.classList.add('active');
  document.getElementById('viewerLabel').textContent = file + ' — page ' + page;
  const viewer = document.getElementById('viewer');
  viewer.src = '';  // force reload so #page= jump always applies
  requestAnimationFrame(() => {
    viewer.src = '/docs/' + encodeURIComponent(file) + '#page=' + page;
  });
}
function addSources(sources) {
  if (!sources || !sources.length) return;
  const wrap = document.createElement('div');
  wrap.className = 'sources';
  sources.forEach(s => {
    const chip = document.createElement('span');
    chip.className = 'src';
    chip.textContent = '📄 ' + s.file.replace(/\\.pdf$/i, '') + ' · p.' + s.page;
    chip.onclick = () => showSource(s.file, s.page, chip);
    wrap.appendChild(chip);
  });
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  // Auto-open the first cited page.
  showSource(sources[0].file, sources[0].page, wrap.firstChild);
}
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('q'), btn = document.getElementById('b');
  const text = input.value.trim();
  if (!text) return;
  input.value = ''; btn.disabled = true;
  messages.push({role: 'user', content: text});
  add('user', text);
  const pending = add('assistant', '…thinking');
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({messages})
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    renderMarkdown(pending, data.reply);
    messages.push({role: 'assistant', content: data.reply});
    addSources(data.sources);
  } catch (err) {
    pending.textContent = '[error] ' + err.message;
  } finally {
    btn.disabled = false;
    input.focus();
  }
});
</script>
</body>
</html>"""


@app.get("/")
def home():
    return HTMLResponse(CHAT_PAGE)


# Serve the PDFs so the viewer can open cited pages. On Vercel the CDN
# usually serves public/ directly; this mount is the fallback.
_PDF_DIR = Path(__file__).parent / "public" / "docs"
if _PDF_DIR.is_dir():
    app.mount("/docs", StaticFiles(directory=_PDF_DIR), name="docs")
