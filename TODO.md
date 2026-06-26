# Product TODO

## Phase 0: Architecture Lock

- [ ] Confirm fixed product scope: enterprise RAG assistant, not workflow builder.
- [ ] Confirm target deployment mode: single company per deployment or multi-tenant SaaS.
- [ ] Confirm file types for v1: PDF, DOCX, XLSX, CSV, TXT, HTML.
- [ ] Confirm LLM provider policy: OpenAI, Claude, local model, or provider abstraction.
- [ ] Confirm data retention and document deletion policy.

## Phase 1: Foundation

- [ ] Implement backend API skeleton.
- [ ] Implement web login and workspace shell.
- [ ] Implement database schema for users, workspaces, documents, chunks, jobs, and conversations.
- [ ] Implement object storage for uploaded files.
- [ ] Implement background job queue.

## Phase 2: Ingestion

- [ ] Build parser registry by file type.
- [ ] Build semantic chunking pipeline.
- [ ] Extract chunk metadata and quality features.
- [ ] Store document processing status.
- [ ] Add retry and dead-letter handling for failed ingestion.

## Phase 3: Retrieval

- [ ] Implement hierarchical domain routing.
- [ ] Implement Elasticsearch or OpenSearch BM25 retrieval.
- [ ] Implement Qdrant vector retrieval.
- [ ] Merge candidates with reciprocal rank fusion.
- [ ] Add BGE or external reranker.
- [ ] Add token-aware context selection.

## Phase 4: Chat

- [ ] Implement grounded generation.
- [ ] Enforce insufficient-context refusal.
- [ ] Add citations.
- [ ] Add conversation history with strict retrieval boundary.
- [ ] Add answer feedback.

## Phase 5: Admin and Operations

- [ ] Admin document dashboard.
- [ ] Processing job dashboard.
- [ ] Retrieval trace viewer.
- [ ] Audit logs.
- [ ] Monitoring and alerting.

## Phase 6: Production

- [ ] Docker Compose for local stack.
- [ ] Production Dockerfiles.
- [ ] Secrets management.
- [ ] Backup and restore plan.
- [ ] Kubernetes manifests or Helm chart.


