# API Surface

This file defines the intended API shape only. Implementation belongs in `apps/api`.

## Auth

```text
POST /auth/login
POST /auth/logout
GET  /auth/me
```

## Workspace

```text
GET  /workspaces/current
GET  /workspaces/current/users
POST /workspaces/current/users/invite
```

## Documents

```text
POST /documents/upload
GET  /documents
GET  /documents/{document_id}
POST /documents/{document_id}/reprocess
DELETE /documents/{document_id}
```

## Chat

```text
POST /chat/sessions
GET  /chat/sessions
POST /chat/sessions/{session_id}/messages
GET  /chat/sessions/{session_id}/trace/{message_id}
```

## Admin

```text
GET /admin/jobs
GET /admin/audit-logs
GET /admin/retrieval-traces
```

