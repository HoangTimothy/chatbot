# Deployment Plan

## Local Development

```text
Web app
API
Worker
Postgres
Redis
Elasticsearch or OpenSearch
Qdrant
Object storage emulator
```

## Production Baseline

```text
Load balancer
  -> Web container
  -> API container
  -> Worker container

Managed Postgres
Managed Redis
Managed object storage
Managed or self-hosted search
Managed or self-hosted vector database
External LLM provider
```

## Future Kubernetes

Use Kubernetes only after the product has stable traffic and operational needs.

Early production can run with Docker Compose or a managed container service.

## Production Concerns

- Secret rotation.
- Per-workspace data isolation.
- Backups for Postgres and object storage.
- Index rebuild strategy.
- LLM provider outage fallback.
- Job retry and dead-letter queue.
- Audit logging.
- Observability and cost tracking.

