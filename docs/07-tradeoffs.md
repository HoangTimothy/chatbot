# Trade-offs

## Fixed Template vs Dify-Style Builder

Fixed template:

- Better for non-IT companies.
- Easier support.
- Lower operational risk.
- More consistent answer quality.

Dify-style builder:

- More flexible.
- Better for technical teams.
- Higher configuration risk.
- Harder to support at scale.

Decision: use a fixed enterprise template for v1.

## Local Models vs API Models

Local models:

- Better data control.
- Higher hardware and DevOps burden.
- Slower on CPU-only machines.

API models:

- Faster to launch.
- Better quality with less infrastructure.
- Requires security review and vendor policy.

Decision: keep provider abstraction, but default to API models for practical v1 delivery.

## Elasticsearch and Qdrant vs Single Vector DB

Hybrid search:

- Better exact term matching.
- Better product code and policy retrieval.
- More infrastructure.

Vector-only:

- Simpler.
- Higher hallucination and recall risk.

Decision: use hybrid search.

