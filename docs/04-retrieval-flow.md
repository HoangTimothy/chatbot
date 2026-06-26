# Retrieval Flow

## Pipeline

```text
Question
  -> Normalize question
  -> Determine workspace and permissions
  -> Route to knowledge branch
  -> Keyword search
  -> Vector search
  -> Merge candidates
  -> Rerank
  -> Select context
  -> Generate grounded answer
```

## Retrieval Priority

Use this order of preference:

1. Hierarchical routing.
2. Hybrid search.
3. Vector embeddings.

Vector search alone is not acceptable for enterprise accuracy.

## Routing

The router should return:

```text
branch_path
confidence
rationale
fallback_allowed
```

If confidence is low, retrieval may use a higher-level branch, but should not search every document by default.

## Hybrid Search

Use both:

```text
BM25 keyword candidates
Vector embedding candidates
```

Then merge with a deterministic method such as reciprocal rank fusion.

## Reranking

Preferred rerankers:

- BGE reranker.
- Cohere rerank.
- Cross encoder model.

Select only the top 3 to 5 chunks for the LLM.

## Grounding Policy

The assistant must answer only using selected chunks.

If evidence is insufficient:

```text
I cannot find sufficient information in the knowledge base.
```

