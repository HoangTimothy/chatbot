"""Ingestion flow skeleton.

This file defines the intended order only.
No parsing, chunking, or indexing logic is implemented here yet.
"""


def ingestion_flow_steps() -> tuple[str, ...]:
    """Return the approved ingestion pipeline order."""
    return (
        "validate_upload",
        "detect_file_type",
        "parse_file",
        "normalize_blocks",
        "detect_document_structure",
        "resolve_knowledge_branch",
        "semantic_chunk",
        "extract_chunk_features",
        "persist_chunks",
        "index_keyword_store",
        "index_vector_store",
        "emit_ingestion_trace",
    )

