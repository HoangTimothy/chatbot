# Product Scope

## Goal

Build a fixed enterprise RAG assistant that lets a company upload knowledge documents and ask grounded questions through a simple web interface.

## Non-Goal

This is not a general workflow builder like Dify.

The system should not expose complex pipeline configuration to normal company users. Architecture is controlled by the product team.

## User Experience

```text
Admin logs in
  -> Creates or enters company workspace
  -> Uploads documents
  -> Reviews processing status
  -> Users ask questions
  -> Answers include citations and confidence signals
```

## Enterprise Template

The product has one default template:

```text
Question
  -> Domain Routing
  -> Hybrid Search
  -> Reranking
  -> Context Selection
  -> Grounded Generation
```

## Why Fixed Architecture

Advantages:

- Easier for non-IT companies.
- Lower chance of bad retrieval configuration.
- Stronger hallucination control.
- Easier support and monitoring.
- Easier compliance review.

Trade-offs:

- Less flexible than Dify.
- Product team must choose good defaults.
- Some advanced users may want customization later.

