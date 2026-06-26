# API Routes

Planned route groups:

```text
auth
workspaces
documents
chat
admin
health
```

Route handlers should:

1. Validate request.
2. Resolve authenticated user and workspace.
3. Call service layer.
4. Return response schema.

Route handlers should not contain RAG logic.

