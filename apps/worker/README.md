# Worker App

Async worker boundary for ingestion, chunking, feature extraction, and indexing.

## Responsibilities

```text
Consume ingestion jobs
Parse uploaded files
Chunk semantically
Extract metadata and quality features
Write chunks to database
Index chunks to keyword and vector stores
Update job status
```

The worker should be idempotent. Re-running a job for the same document version should not create duplicate chunks.

