"""Main evaluation runner — orchestrates the full evaluation pipeline.

Flow:
1. Load golden dataset
2. For each sample → run RAG pipeline (retrieve → context select → generate)
3. Compute Tier 1 metrics (deterministic)
4. Compute Tier 2 metrics (RAGAS, if enabled)
5. Compute Tier 3 metrics (LLM judge, if enabled)
6. Print report and save JSON
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Ensure project root is on the path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from evaluation.config import EvalConfig
from evaluation.dataset import EvalResult, EvalSample, load_dataset
from evaluation.metrics.retrieval_metrics import (
    aggregate_retrieval_metrics,
    compute_retrieval_metrics,
)
from evaluation.metrics.generation_metrics import (
    aggregate_generation_metrics,
    compute_generation_metrics,
)
from evaluation.reporters.console_reporter import (
    print_footer,
    print_header,
    print_per_sample_detail,
    print_tier1_results,
    print_tier2_results,
    print_tier3_results,
    save_json_report,
)

logger = logging.getLogger("evaluation.runner")


# ──────────────────────────────────────────────────────────────────────
# RAG Pipeline adapter
# ──────────────────────────────────────────────────────────────────────


def _build_rag_pipeline(config: EvalConfig):
    """Build the RAG pipeline from existing project components.

    Returns a ChatPipeline instance or None if dependencies are unavailable.
    """
    try:
        # Add package paths so imports work
        packages_path = Path(__file__).resolve().parent.parent / "packages"
        for pkg_dir in ["rag_core", "shared"]:
            pkg_path = str(packages_path / pkg_dir)
            if pkg_path not in sys.path:
                sys.path.insert(0, pkg_path)

        apps_path = str(Path(__file__).resolve().parent.parent / "apps" / "api")
        if apps_path not in sys.path:
            sys.path.insert(0, apps_path)

        from rag_core.flows.chat_flow import ChatPipeline
        from rag_core.flows.retrieval_flow import RetrievalPipeline

        # Attempt to import factory/wiring from the API app
        # Fall back to building pipeline from adapters directly
        from shared.models import Base

        # Import adapters
        from rag_core.adapters.generator import OpenAIGenerator
        from rag_core.adapters.reranker import RerankerAdapter
        from rag_core.adapters.searchers import (
            ElasticsearchSearcher,
            HydeVectorSearcher,
            QdrantSearcher,
            SQLDatabaseSearcher,
        )
        from rag_core.services.context_selector import TokenAwareContextSelector
        from rag_core.services.router import DomainRouter

        # Read config from environment
        from dotenv import load_dotenv
        load_dotenv()

        openai_key = os.getenv("OPENAI_API_KEY", "")
        google_key = os.getenv("GOOGLE_API_KEY", "")
        es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        embedding_model = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
        llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        llm_provider = os.getenv("LLM_PROVIDER", "openai")

        # Database session for SQL fallback
        db_url = os.getenv("DATABASE_URL", "")
        db_session_factory = None

        if db_url:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            engine = create_engine(db_url)
            db_session_factory = sessionmaker(bind=engine)
        else:
            # Try SQLite fallback
            db_path = Path(__file__).resolve().parent.parent / "rag.db"
            if db_path.exists():
                from sqlalchemy import create_engine
                from sqlalchemy.orm import sessionmaker
                engine = create_engine(f"sqlite:///{db_path}")
                db_session_factory = sessionmaker(bind=engine)

        sql_searcher = None
        if db_session_factory:
            sql_searcher = SQLDatabaseSearcher(db_session_factory=db_session_factory)

        # Build pipeline components
        router = DomainRouter(available_branches=[], openai_api_key=openai_key, model_name=llm_model)
        keyword_searcher = ElasticsearchSearcher(
            es_url=es_url, fallback_searcher=sql_searcher
        )

        base_vector_searcher = QdrantSearcher(
            qdrant_url=qdrant_url,
            embedding_model=embedding_model,
            openai_api_key=openai_key,
            google_api_key=google_key,
            fallback_searcher=sql_searcher,
        )

        generator = OpenAIGenerator(
            openai_api_key=openai_key,
            model_name=llm_model,
        )

        # Wrap with HyDE
        if config.hyde_enabled:
            logger.info("HyDE is enabled for evaluation vector search.")
            vector_searcher = HydeVectorSearcher(
                vector_searcher=base_vector_searcher,
                generator=generator,
            )
        else:
            logger.info("HyDE is disabled for evaluation vector search.")
            vector_searcher = base_vector_searcher

        reranker = RerankerAdapter(provider="local")
        context_selector = TokenAwareContextSelector(max_tokens=4000)

        retrieval_pipeline = RetrievalPipeline(
            router=router,
            keyword_searcher=keyword_searcher,
            vector_searcher=vector_searcher,
            reranker=reranker,
        )

        chat_pipeline = ChatPipeline(
            retrieval_pipeline=retrieval_pipeline,
            context_selector=context_selector,
            generator=generator,
        )

        logger.info("RAG pipeline built successfully for evaluation.")
        return chat_pipeline

    except Exception as e:
        logger.warning(f"Could not build RAG pipeline: {e}. Running in offline/mock mode.")
        return None


def _run_rag_pipeline(
    pipeline,
    sample: EvalSample,
    workspace_id: str,
    search_limit: int,
    rerank_limit: int,
) -> EvalResult:
    """Execute the RAG pipeline for a single sample and capture outputs."""
    result = EvalResult(sample=sample)

    if pipeline is None:
        # Offline mode: return empty result so metrics still compute (zeros)
        return result

    try:
        # Measure retrieval + generation time together
        start_time = time.perf_counter()

        routed_question, candidates, selected_context, grounded_answer = (
            pipeline.generate_response(
                workspace_id=workspace_id,
                question=sample.question,
                search_limit=search_limit,
                rerank_limit=rerank_limit,
            )
        )

        total_ms = (time.perf_counter() - start_time) * 1000

        # Populate result fields
        result.retrieved_chunk_ids = [c.chunk.chunk_id for c in candidates]
        result.retrieved_texts = [c.chunk.text for c in candidates]
        result.retrieval_latency_ms = total_ms * 0.6  # approximate split

        result.generated_answer = grounded_answer.answer
        result.generated_citations = list(grounded_answer.citations)
        result.predicted_insufficient = grounded_answer.insufficient_context
        result.generation_latency_ms = total_ms * 0.4

        # Build context text string for the judge
        result.context_text = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.text}" for chunk in selected_context.chunks
        )

    except Exception as e:
        logger.error(f"Pipeline execution failed for '{sample.question[:50]}': {e}")

    return result


# ──────────────────────────────────────────────────────────────────────
# Main evaluation orchestrator
# ──────────────────────────────────────────────────────────────────────


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    """Execute the full evaluation pipeline.

    Args:
        config: Evaluation configuration.

    Returns:
        Dictionary with all aggregated scores across tiers.
    """
    # 1. Load dataset
    samples = load_dataset(config.dataset_path, max_samples=config.max_samples)
    print_header(config.dataset_path, len(samples), config.tiers)

    # 2. Build RAG pipeline
    pipeline = None
    if any(t in config.tiers for t in [1, 2, 3]):
        pipeline = _build_rag_pipeline(config)

    # 3. Run pipeline for each sample
    workspace_id = os.getenv("EVAL_WORKSPACE_ID", "default")
    results: list[EvalResult] = []

    for i, sample in enumerate(samples):
        logger.info(f"Processing sample {i + 1}/{len(samples)}: {sample.question[:60]}...")
        result = _run_rag_pipeline(
            pipeline=pipeline,
            sample=sample,
            workspace_id=workspace_id,
            search_limit=config.retrieval_k,
            rerank_limit=config.rerank_limit,
        )
        results.append(result)

    # 4. Compute metrics per tier
    aggregated: dict[str, Any] = {}

    # Tier 1: Deterministic component metrics
    if 1 in config.tiers:
        logger.info("Computing Tier 1 — Component Metrics...")
        for r in results:
            compute_retrieval_metrics(r, k=config.retrieval_k)
            compute_generation_metrics(r)

        retrieval_agg = aggregate_retrieval_metrics(results)
        generation_agg = aggregate_generation_metrics(results)

        aggregated["tier1_retrieval"] = retrieval_agg
        aggregated["tier1_generation"] = generation_agg

        print_tier1_results(retrieval_agg, generation_agg)

    # Tier 2: RAGAS
    if 2 in config.tiers:
        logger.info("Computing Tier 2 — RAGAS Metrics...")
        from evaluation.metrics.ragas_metrics import compute_ragas_metrics, is_available

        if is_available() or config.ragas_enabled:
            ragas_scores = compute_ragas_metrics(results)
            aggregated["tier2_ragas"] = ragas_scores
            print_tier2_results(ragas_scores)
        else:
            logger.warning("RAGAS not available. Skipping Tier 2.")
            print("\n  [WARN] Tier 2 skipped: RAGAS not installed (pip install ragas datasets)")

    # Tier 3: LLM Judge
    if 3 in config.tiers:
        logger.info("Computing Tier 3 — LLM Judge Metrics...")
        from evaluation.metrics.llm_judge_metrics import compute_judge_metrics

        judge_scores = compute_judge_metrics(
            results=results,
            provider=config.llm_judge_provider,
            model=config.llm_judge_model,
            api_key=config.llm_judge_api_key,
            prompt_path=config.judge_prompt_path,
        )
        aggregated["tier3_judge"] = judge_scores
        print_tier3_results(judge_scores)

    # 5. Per-sample details (if verbose)
    if config.verbose:
        print("\n  == Per-Sample Details ==")
        for i, r in enumerate(results):
            print_per_sample_detail(r, i)

    # 6. Save JSON report
    report_path = save_json_report(
        output_dir=config.output_dir,
        results=results,
        aggregated=aggregated,
        dataset_path=config.dataset_path,
        tiers=config.tiers,
    )
    aggregated["report_path"] = report_path

    print_footer()
    return aggregated


# Allow direct execution for quick testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    config = EvalConfig(tiers=[1])
    run_evaluation(config)
