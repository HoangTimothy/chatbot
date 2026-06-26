# Data Flow

## Upload Flow

```text
User
  -> Web upload
  -> API validates file and workspace
  -> Store original file
  -> Create document record
  -> Enqueue ingestion job
  -> Worker parses file
  -> Worker chunks content
  -> Worker extracts metadata and features
  -> Worker indexes chunks
  -> API shows processing status
```

## Chat Flow

```text
User question
  -> Auth and workspace validation
  -> Domain routing
  -> Hybrid retrieval inside workspace and routed branch
  -> Reranking
  -> Context selection
  -> Grounded LLM generation
  -> Citation assembly
  -> Conversation and trace logging
```

## Core Data Entities

```text
Workspace
User
Role
Document
DocumentVersion
Chunk
IngestionJob
KnowledgeBranch
Conversation
Message
RetrievalTrace
AnswerFeedback
AuditLog
```

## Workspace Isolation

Every query and index operation must include:

```text
workspace_id
document_visibility
user_role
knowledge_branch
```

No retrieval operation should search across all company data unless explicitly authorized.

