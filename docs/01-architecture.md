# Architecture

## System Diagram

```text
Browser
  |
  v
Web App
  |
  v
API Gateway / Backend API
  |
  +---------------- Auth and Workspace
  |
  +---------------- Document Management
  |                   |
  |                   v
  |                 Object Storage
  |
  +---------------- Chat API
  |                   |
  |                   v
  |                 RAG Core
  |
  +---------------- Admin API
                      |
                      v
                    Audit Logs

Worker
  |
  +---------------- Parse Documents
  +---------------- Semantic Chunking
  +---------------- Metadata and Feature Extraction
  +---------------- Keyword Index
  +---------------- Vector Index

Datastores
  |
  +---------------- Postgres
  +---------------- Redis / Queue
  +---------------- Object Storage
  +---------------- Elasticsearch or OpenSearch
  +---------------- Qdrant
```

## Main Services

### Web App

Responsible for login, upload, chat, admin dashboard, document status, and trace views.

### API

Responsible for request validation, auth, workspace isolation, document metadata, chat orchestration, and admin APIs.

### Worker

Responsible for asynchronous file processing, chunking, feature extraction, and indexing.

### RAG Core

Shared package that defines retrieval and generation flows. It must stay independent from FastAPI, UI, database, and queue details.

## Dependency Direction

```text
apps/api      -> packages/rag_core
apps/worker   -> packages/rag_core
apps/web      -> packages/shared
packages/*    -> no dependency on apps/*
```

## Production Principle

Keep business logic out of the web UI and route handlers. Put orchestration in services and pure RAG decisions in `packages/rag_core`.

