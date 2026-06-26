# Auth and Tenancy

## Roles

```text
Owner
Admin
Knowledge Manager
Member
Read-only Auditor
```

## Required Capabilities

### Owner

- Manage billing or deployment config.
- Manage workspace settings.
- Manage admins.

### Admin

- Invite users.
- Manage document visibility.
- View audit logs.

### Knowledge Manager

- Upload documents.
- Reprocess documents.
- Review ingestion warnings.

### Member

- Ask questions.
- View allowed citations.
- Submit feedback.

### Auditor

- View logs and traces.
- Cannot upload or ask private questions unless authorized.

## Security Rules

- Every request must include workspace context.
- Every document must have visibility rules.
- Every retrieval query must enforce workspace and role filters.
- API keys must never be exposed to the web app.
- Raw documents should not be committed to Git.

