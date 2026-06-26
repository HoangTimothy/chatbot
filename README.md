# Enterprise RAG Platform

This repository contains the complete implementation of a production-grade, enterprise-ready knowledge assistant. It features an end-to-end RAG (Retrieval-Augmented Generation) template that minimizes manual pipeline configuration while delivering maximum retrieval precision and grounding reliability.

## Core Features

- **Intelligent Query Routing**: Classifies incoming questions via LLM into appropriate retrieval strategies (`kb_only`, `web_search`, `knowledge_graph`, `kb_and_web`, `kb_and_kg`, or `all`) and targets matching knowledge branches.
- **Multi-Source Retrieval & Fusion**: Merges internal vector/keyword search with external web search (Tavily API) and entity relationships (NetworkX in-memory Knowledge Graph) using generalized Reciprocal Rank Fusion (RRF).
- **Contextual Ingestion**: Situates raw document chunks in their parent document's context during ingestion for significantly higher vector match rates.
- **HyDE (Hypothetical Document Embeddings)**: Optional query transformation module that generates candidate answers to improve dense retrieval accuracy.
- **Grounded Generation & Citations**: Synthesizes structured markdown responses strictly grounded on retrieved sources, complete with source-type citation badges (`📄 KB Chunk`, `🌐 Web Result`, `🔗 KG Triplet`).
- **HEX / Clean Architecture**: Implements a strict port-and-adapter hexagonal layout separating core business logic (`packages/rag_core`) from external API gateways and services (`apps/`).

---

## Repository Layout

```text
apps/
  api/          Backend API (FastAPI) orchestrating routes and trace logs.
  web/          Web application frontend.
  worker/       Async ingestion and indexing worker processing document queues.

packages/
  rag_core/     Shared core RAG services, pipelines, contracts, and adapters.
  shared/       Database models, database schemas, and shared enums.

config/
  Fixed enterprise template and folder structure configs.

prompts/
  System instructions for query routing, entity extraction, HyDE, and grounding.

tests/
  Unit and integration test suites covering all routing and pipeline flows.
```

---

## Installation & Setup

1. **Python Dependencies**:
   Install all package requirements:
   ```bash
   pip install fastapi sqlalchemy alembic uvicorn pydantic-settings tiktoken aiofiles pypdf qdrant-client elasticsearch tavily-python python-multipart bcrypt aiosqlite pytest requests-mock Faker
   ```

2. **Environment Configuration**:
   Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```
   Fill in your LLM provider (`google` or `openai`), model names, and corresponding API keys (e.g., `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `TAVILY_API_KEY`).

3. **Running the Services**:
   - **Backend API**:
     ```bash
     uvicorn apps.api.app.main:app --host 127.0.0.1 --port 8000 --reload
     ```
   - **Ingestion Worker**:
     ```bash
     python apps/worker/worker/main.py
     ```
   - **Frontend App**:
     ```bash
     # From apps/web directory
     npm run dev
     ```

---

## Verification & Testing

Verify code correctness and backward compatibility using the test suite:
```bash
python -m pytest
```

All 25 tests covering query routing, web searching, knowledge graph traversals, semantic chunking, and database persistence must pass successfully.
