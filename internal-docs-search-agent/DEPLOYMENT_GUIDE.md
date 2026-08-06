# Deploying the Internal Docs Search Agent to Vercel

A start-to-finish guide for turning the local `langgraph dev` RAG agent
(`internal_docs_search_agent.py`) into a public web app on Vercel, with a chat
UI that shows the cited PDF page next to each answer. This was done once on
2026-08-06 and then torn down; follow this to recreate it.

## Architecture

```
docs/*.pdf ──▶ scripts/build_index.py ──▶ index_data.json   (offline, run locally)
                                              │
                                              ▼
                    main.py  =  retrieval tool + create_agent + FastAPI + chat UI
                                              │
                                              ▼
                    Vercel:  serverless function (all routes) + CDN (public/docs/*.pdf)
```

**The core idea:** Vercel serverless can't run `langgraph dev` or hold a Chroma
DB, so the vector index is pre-built locally into a JSON file (chunks +
embeddings). At runtime the function loads the JSON, embeds only the query, and
does cosine similarity with numpy. The agent itself (`create_agent`) is
unchanged from the local version.

## File layout (final state)

```
internal-docs-search-agent/
├── internal_docs_search_agent.py   # original langgraph dev version (untouched)
├── main.py                         # Vercel entrypoint: agent + FastAPI + embedded chat UI
├── index_data.json                 # pre-built index (generated, ~2.5MB)
├── requirements.txt                # runtime deps only
├── scripts/build_index.py          # offline indexer (uv inline-metadata script)
├── docs/*.pdf                      # source PDFs (indexing input)
├── public/docs/*.pdf               # same PDFs, copied here so the CDN serves them
├── .vercelignore                   # keeps .env, docs/, scripts/, langgraph files out
└── .env                            # OPENAI_API_KEY etc. (never deployed)
```

## Step 1 — Pre-build the index

`scripts/build_index.py` (uv inline metadata; deps: `langchain-community`,
`langchain-openai`, `langchain-text-splitters`, `pypdf`, `python-dotenv`):

1. `PyPDFLoader` every PDF in `docs/`
2. `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
3. `OpenAIEmbeddings(model="text-embedding-3-small").embed_documents(...)`
4. Write `index_data.json`: `[{text, source, page, embedding}, ...]`

```bash
uv run scripts/build_index.py     # rerun whenever the PDFs change
cp docs/*.pdf public/docs/        # keep the viewer copies in sync
```

## Step 2 — main.py (the whole app in one file)

Four sections, in order:

**Retrieval tool** — load `index_data.json` at import time into a numpy matrix;
the `@tool search_documents(query)` embeds the query and returns the top-4
chunks formatted as `[Source: <file>, page <n>]\n<text>` (the citation format
matters — the API parses it back out later).

```python
q = np.array(_embeddings.embed_query(query))
sims = _VECTORS @ q / (_NORMS * np.linalg.norm(q))
top = np.argsort(sims)[-4:][::-1]
```

**Agent** — identical to the local version:

```python
agent = create_agent(
    # Reasoning models need reasoning_effort="none" to use function tools
    # on the /v1/chat/completions endpoint.
    model=init_chat_model("openai:gpt-5.6-luna", reasoning_effort="none"),
    tools=[search_documents],
    system_prompt=SYSTEM_PROMPT,   # search first, cite (doc, page), admit gaps
)
```

No checkpointer: the app is stateless; the browser keeps the message history
and POSTs the full list every turn.

**API** — `POST /api/chat` takes `{messages: [{role, content}, ...]}`, runs
`agent.invoke`, and returns `{reply, sources}`. Sources are recovered by
regexing `\[Source: (.+?), page (\d+)\]` out of the **tool messages after the
last human message** (= this turn's retrievals only), deduped. PyPDF pages are
0-indexed, browser `#page=` is 1-indexed → return `page + 1`.

**UI** — one HTML string served at `GET /`. Two panes:
- *Chat*: vanilla JS; assistant replies rendered as markdown with
  syntax-highlighted code via CDN scripts in `<head>` — marked (parse),
  DOMPurify (sanitize the HTML before injection), highlight.js core +
  `github-dark.min.css` (all from jsdelivr). Fall back to `textContent` if
  `window.marked` is missing. After injecting:
  `el.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b))`.
- *Viewer*: each answer's sources render as clickable chips; a chip sets an
  iframe to `/docs/<file>#page=<n>` — the browser's native PDF viewer jumps to
  the page. Reset `iframe.src = ''` before setting it, or repeated `#page=`
  jumps don't apply. Auto-open the first source per answer.

Plus a `StaticFiles` mount of `public/docs` at `/docs` as fallback (on Vercel
the CDN normally serves `public/` before the function is hit).

**requirements.txt** (runtime only — no chromadb, no pypdf, no tavily):

```
fastapi
langchain>=1.0
langchain-openai
numpy
```

## Step 3 — Deploy

```bash
vercel link --yes --project internal-docs-search-agent

# add the key without echoing it to the terminal
python3 -c "import re
for line in open('.env'):
    m = re.match(r'OPENAI_API_KEY=[\"\\']?([^\"\\'\n]+)', line)
    if m: print(m.group(1), end='')" | vercel env add OPENAI_API_KEY production

vercel deploy --prod --yes
```

Then **disable Deployment Protection** (on by default, blocks anonymous
visitors with Vercel SSO): dashboard → Project → Settings → Deployment
Protection → Vercel Authentication off. (Or via the Vercel MCP
`update_project_deployment_protection` with `ssoProtection: {enabled: false}`.)

Verify:

```bash
curl -s -o /dev/null -w "%{http_code}" https://<project>.vercel.app/            # 200
curl -s -o /dev/null -w "%{content_type}" https://<project>.vercel.app/docs/<file>.pdf   # application/pdf
curl -s -X POST https://<project>.vercel.app/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is a vector store?"}]}'
```

## Gotchas (each one cost a debugging round)

1. **Do NOT use a catch-all rewrite** (`{"source": "/(.*)", "destination": "/api/index"}`
   with the app in `api/index.py`). Vercel now forwards the *rewritten* path to
   the app, so FastAPI receives `/api/index` for every request and all routes
   404. **Fix:** app in a root-level `main.py` with `app = FastAPI()` — Vercel
   auto-detects FastAPI and routes *all* paths to it. No `vercel.json` at all.
2. **Deployment Protection 401s everything by default** — anonymous curl gets
   `{"error": {"code": "401"}}` / a 302 to SSO. Disable it (see above) for a
   public app. Note the tradeoff: `/api/chat` then burns your OpenAI key for
   anyone with the URL.
3. **Chroma doesn't belong in serverless** — chromadb's size and sqlite
   requirements fight the 250MB bundle limit and cold starts. Pre-computing the
   index kills the dependency entirely.
4. **Page numbering** is 0-indexed in PyPDF metadata, 1-indexed in `#page=`.
5. **`docs` in `.vercelignore` also matches `public/docs`** (gitignore
   semantics). Anchor it: `/docs`.
6. **Function timeout:** agent turns with tool calls can exceed the old 10s
   default. With Fluid compute (current default) it's fine; if you ever need
   it explicit, that's the one reason to add a `vercel.json`
   (`functions.main.py.maxDuration`).
7. The `reasoning_effort="none"` on `init_chat_model` is required for the
   reasoning model to use function tools on `/v1/chat/completions`.

## Tearing it down

```bash
vercel remove internal-docs-search-agent --yes   # deletes the project: deployments, alias, env vars
rm -rf .vercel                                   # remove the local link
# (note: `vercel project rm` does NOT accept --yes / -y in CLI 56; use `vercel remove`)
```
