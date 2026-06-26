"""Flow-only worker entrypoint.

TODO:
    Implement queue consumer and job handlers in the coding phase.
"""


def ingestion_worker_steps() -> tuple[str, ...]:
    """Return the intended ingestion worker step order."""
    return (
        "load_job",
        "validate_workspace_and_document",
        "download_original_file",
        "parse_file",
        "chunk_document",
        "extract_chunk_features",
        "persist_chunks",
        "index_keyword_store",
        "index_vector_store",
        "mark_job_completed",
    )

