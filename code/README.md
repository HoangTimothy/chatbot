# Enterprise Knowledge Assistant

Production-oriented RAG service scaffold for internal company documents.

## Architecture

```text
Question
  |
  v
Domain Routing
  |
  v
Hybrid Search
  |-- Elasticsearch / BM25
  |-- Qdrant / Vector embeddings
  v
Reranking
  |
  v
Context Selection
  |
  v
Grounded Generation
```

The service is designed to avoid broad corpus search by routing each query to a likely document branch first. Retrieval uses hybrid keyword and vector search, then reranking selects a compact evidence set before generation.

## Folder Layout

```text
app/
  api/            FastAPI routes
  core/           settings, logging, shared errors
  generation/     grounded answer generation ports
  ingestion/      document loading and semantic chunking
  reranking/      reranker interfaces and defaults
  retrieval/      hybrid search, Qdrant, Elasticsearch adapters
  routing/        hierarchical domain router
  schemas/        request/response and retrieval models
  services/       RAG orchestration
tests/            focused unit tests
docker/           container assets
```

## Local Run

```bash
cd code
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

On Windows PowerShell:

```powershell
cd code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

## Production Wiring

Set these environment variables before enabling real retrieval and generation:

```text
ELASTICSEARCH_URL=http://localhost:9200
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=enterprise_chunks
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
LLM_PROVIDER=openai
OPENAI_API_KEY=...
```

## Refusal Policy

The assistant must answer only from retrieved context. If routing, retrieval, reranking, or context selection cannot provide enough evidence, the response must be:

```text
I cannot find sufficient information in the knowledge base.
```

