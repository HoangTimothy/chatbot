import os
import logging
from typing import Sequence, Any
import random
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from sqlalchemy import select, or_
from sqlalchemy.orm import Session

from rag_core.contracts.types import Chunk, DocumentRef, RetrievalCandidate, RoutedQuestion
from rag_core.ports.interfaces import KeywordSearchPort, VectorSearchPort
from shared.models import Chunk as DbChunk

logger = logging.getLogger("rag_core.searchers")


def _db_chunk_to_contract(db_chunk: DbChunk) -> Chunk:
    """Helper to convert SQLAlchemy DB model back into RAG core Chunk contract dataclass."""
    # Convert slash-delimited string path back to tuple
    branch_path = tuple(db_chunk.knowledge_branch_path.split("/")) if db_chunk.knowledge_branch_path else ()
    
    doc_ref = DocumentRef(
        workspace_id=db_chunk.workspace_id,
        document_id=db_chunk.document_id,
        document_version_id=db_chunk.document_version_id,
        file_name=db_chunk.source_file_name,
        file_hash=db_chunk.source_file_hash
    )
    
    features = {
        "page_number": db_chunk.page_number,
        "sheet_name": db_chunk.sheet_name,
        "section_title": db_chunk.section_title,
        "char_count": db_chunk.char_count,
        "table_count": db_chunk.table_count,
        "image_count": db_chunk.image_count,
        "contains_policy_language": db_chunk.contains_policy_language,
        "contains_product_spec": db_chunk.contains_product_spec,
        "contains_procedure_steps": db_chunk.contains_procedure_steps,
        "chunk_quality_score": db_chunk.chunk_quality_score,
        "chunking_strategy": db_chunk.chunking_strategy,
        "chunk_version": db_chunk.chunk_version
    }
    
    # Filter out None values in features
    features = {k: v for k, v in features.items() if v is not None}
    
    return Chunk(
        chunk_id=db_chunk.id,
        document=doc_ref,
        text=db_chunk.text,
        token_count=db_chunk.token_count,
        knowledge_branch_path=branch_path,
        features=features
    )


class SQLDatabaseSearcher(KeywordSearchPort, VectorSearchPort):
    """Fallback searcher that queries SQL database directly when search clusters are offline."""

    def __init__(self, db_session_factory: Any):
        self.db_session_factory = db_session_factory

    def search(self, routed_question: RoutedQuestion, limit: int) -> Sequence[RetrievalCandidate]:
        """Runs search on the SQL database using string match heuristics.

        Satisfies both search port interfaces for easy fallback.
        """
        logger.info("Executing database-backed SQL fallback search...")
        db: Session = self.db_session_factory()
        try:
            # Base query filtered by workspace
            base_stmt = select(DbChunk).where(DbChunk.workspace_id == routed_question.workspace_id)

            # Try branch-filtered query first, fall back to full workspace search if 0 results
            db_chunks = []
            if routed_question.branch_path:
                branch_str = "/".join(routed_question.branch_path)
                # Use prefix match: branch 'general' also matches 'general/faq', etc.
                branch_stmt = base_stmt.where(
                    or_(
                        DbChunk.knowledge_branch_path == branch_str,
                        DbChunk.knowledge_branch_path.like(f"{branch_str}/%"),
                    )
                )
                branch_result = db.execute(branch_stmt)
                db_chunks = branch_result.scalars().all()
                if not db_chunks:
                    logger.warning(
                        f"Branch filter '{branch_str}' returned 0 chunks. "
                        "Falling back to workspace-wide search."
                    )

            # Workspace-wide fallback when no branch filter or branch returned nothing
            if not db_chunks:
                result = db.execute(base_stmt)
                db_chunks = result.scalars().all()

            # Rank items based on keyword term frequency overlap (simple TF/Overlap scorer)
            query_terms = set(routed_question.question.lower().split())
            candidates = []
            
            for db_chunk in db_chunks:
                text_lower = db_chunk.text.lower()
                # Count matching terms
                match_count = sum(1 for term in query_terms if term in text_lower)
                
                # Heuristic score between 0.0 and 1.0
                score = match_count / max(len(query_terms), 1)
                
                # Only return candidates with at least 1 keyword match
                if score > 0:
                    contract_chunk = _db_chunk_to_contract(db_chunk)
                    candidates.append(
                        RetrievalCandidate(
                            chunk=contract_chunk,
                            score=score,
                            source="sql_database"
                        )
                    )
            
            # Sort descending by score
            candidates.sort(key=lambda x: x.score, reverse=True)
            return candidates[:limit]

        finally:
            db.close()


class ElasticsearchSearcher(KeywordSearchPort):
    """Keyword BM25 searcher adapter connecting to Elasticsearch."""

    def __init__(self, es_url: str, index_name: str = "enterprise_chunks", fallback_searcher: SQLDatabaseSearcher | None = None):
        self.es_url = es_url
        self.index_name = index_name
        self.fallback_searcher = fallback_searcher
        self.client = None
        self._is_enabled = False
        self.had_fallback = False
        self.fallback_reason = None
        self.init_error = None

        try:
            self.client = Elasticsearch(self.es_url)
            if self.client.ping():
                self._is_enabled = True
            else:
                self.init_error = "Elasticsearch ping returned False"
                logger.warning(f"Elasticsearch ping failed. Using SQL fallback for keywords.")
        except Exception as e:
            self.init_error = f"Connection failed: {str(e)}"
            logger.warning(f"Failed to connect to Elasticsearch: {e}. Using SQL fallback for keywords.")

    def search(self, routed_question: RoutedQuestion, limit: int) -> Sequence[RetrievalCandidate]:
        if not self._is_enabled or not self.client:
            self.had_fallback = True
            self.fallback_reason = self.init_error or "Elasticsearch is disabled"
            if self.fallback_searcher:
                return self.fallback_searcher.search(routed_question, limit)
            return []

        # Construct ES query with filters
        must_queries = [
            {"match": {"text": routed_question.question}},
            {"term": {"workspace_id": routed_question.workspace_id}}
        ]
        
        if routed_question.branch_path:
            must_queries.append({"term": {"knowledge_branch_path": list(routed_question.branch_path)}})

        query_body = {
            "query": {
                "bool": {
                    "must": must_queries
                }
            },
            "size": limit
        }

        try:
            response = self.client.search(index=self.index_name, body=query_body)
            hits = response["hits"]["hits"]
            
            candidates = []
            for hit in hits:
                source = hit["_source"]
                branch_path = tuple(source.get("knowledge_branch_path", []))
                
                doc_ref = DocumentRef(
                    workspace_id=source["workspace_id"],
                    document_id=source["document_id"],
                    document_version_id=source["document_version_id"],
                    file_name=source["source_file_name"],
                    file_hash=""  # not indexed in ES, can remain empty for retrieval
                )
                
                # Fetch quality signals
                features = {
                    "char_count": source.get("char_count"),
                    "table_count": source.get("table_count"),
                    "contains_policy_language": source.get("contains_policy_language"),
                    "contains_product_spec": source.get("contains_product_spec"),
                    "contains_procedure_steps": source.get("contains_procedure_steps"),
                }
                features = {k: v for k, v in features.items() if v is not None}

                chunk = Chunk(
                    chunk_id=hit["_id"],
                    document=doc_ref,
                    text=source["text"],
                    token_count=source.get("token_count", 0),
                    knowledge_branch_path=branch_path,
                    features=features
                )
                
                candidates.append(
                    RetrievalCandidate(
                        chunk=chunk,
                        score=hit["_score"],
                        source="elasticsearch"
                    )
                )
            return candidates

        except Exception as e:
            self.had_fallback = True
            self.fallback_reason = f"Keyword search failed: {str(e)}"
            logger.error(f"Elasticsearch search failed: {e}. Falling back to DB search.")
            if self.fallback_searcher:
                return self.fallback_searcher.search(routed_question, limit)
            raise


class QdrantSearcher(VectorSearchPort):
    """Vector distance searcher adapter connecting to Qdrant.

    Supports both Google (text-embedding-004, dim=768) and OpenAI embedding
    providers. The active provider is controlled by the EMBEDDING_PROVIDER
    environment variable ("google" | "openai").
    """

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str = "enterprise_chunks",
        embedding_model: str = "models/text-embedding-004",
        openai_api_key: str = "",
        google_api_key: str = "",
        fallback_searcher: SQLDatabaseSearcher | None = None,
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.fallback_searcher = fallback_searcher
        self._genai_configured = False
        # Determine embedding provider and dimension
        self.provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
        if self.provider == "google":
            if "gemini-embedding" in embedding_model:
                self.dimension = 3072
            else:
                self.dimension = 768  # text-embedding-004 default output dimension
        else:
            self.dimension = 3072 if "large" in embedding_model else 1536

        self.client = None
        self._is_enabled = False
        self.had_fallback = False
        self.fallback_reason = None
        self.init_error = None

        # Initialise provider-specific clients
        self.openai_client = None
        if self.provider != "google":
            if openai_api_key:
                self.openai_client = OpenAI(api_key=openai_api_key)

        if self.provider == "google" and google_api_key:
            os.environ.setdefault("GOOGLE_API_KEY", google_api_key)

        try:
            self.client = QdrantClient(url=self.qdrant_url)
            self.client.get_collections()
            self._is_enabled = True
        except Exception as e:
            self.init_error = f"Connection failed: {str(e)}"
            logger.warning(f"Failed to connect to Qdrant: {e}. Using SQL fallback for vectors.")

    def _get_embedding(self, text: str) -> list[float]:
        """Fetch embedding from the configured provider (Google or OpenAI)."""
        try:
            if self.provider == "google":
                import google.generativeai as genai
                if not self._genai_configured:
                    api_key = os.getenv("GOOGLE_API_KEY", "")
                    if not api_key:
                        raise ValueError("GOOGLE_API_KEY is not set.")
                    genai.configure(api_key=api_key)
                    self._genai_configured = True

                response = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_query",
                    request_options={"timeout": 30.0}
                )
                return response["embedding"]
            else:
                if not self.openai_client:
                    self.had_fallback = True
                    self.fallback_reason = "Embedding failed: OpenAI API key is missing"
                    random.seed(hash(text))
                    return [random.uniform(-1.0, 1.0) for _ in range(self.dimension)]

                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                )
                return response.data[0].embedding
        except Exception as e:
            self.had_fallback = True
            self.fallback_reason = f"Embedding generation failed ({self.provider}): {str(e)}"
            logger.warning(f"Embedding failed ({self.provider}): {e}. Falling back to mock vector.")
            random.seed(hash(text))
            return [random.uniform(-1.0, 1.0) for _ in range(self.dimension)]

    def search(self, routed_question: RoutedQuestion, limit: int) -> Sequence[RetrievalCandidate]:
        if not self._is_enabled or not self.client:
            self.had_fallback = True
            self.fallback_reason = self.init_error or "Qdrant is disabled"
            if self.fallback_searcher:
                return self.fallback_searcher.search(routed_question, limit)
            return []

        # 1. Embed query text
        query_vector = self._get_embedding(routed_question.question)

        # 2. Setup metadata filters — only filter by workspace_id
        # Branch filtering is unreliable in Qdrant when the field is stored as a plain string
        # (MatchValue with a list doesn't match scalar string fields).
        # Correct branch filtering is delegated to post-retrieval reranking.
        must_conditions = [
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value=routed_question.workspace_id)
            )
        ]

        qdrant_filter = Filter(must=must_conditions)

        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
                limit=limit
            )

            candidates = []
            for hit in hits:
                payload = hit.payload
                branch_path = tuple(payload.get("knowledge_branch_path", []))

                doc_ref = DocumentRef(
                    workspace_id=payload["workspace_id"],
                    document_id=payload["document_id"],
                    document_version_id=payload["document_version_id"],
                    file_name=payload["source_file_name"],
                    file_hash=""
                )

                # Clean features from payload
                features = {}
                feature_keys = [
                    "char_count", "table_count", "image_count",
                    "contains_policy_language", "contains_product_spec",
                    "contains_procedure_steps", "chunk_quality_score"
                ]
                for fk in feature_keys:
                    if fk in payload:
                        features[fk] = payload[fk]

                chunk = Chunk(
                    chunk_id=str(hit.id),
                    document=doc_ref,
                    text=payload["text"],
                    token_count=payload.get("token_count", 0),
                    knowledge_branch_path=branch_path,
                    features=features
                )

                candidates.append(
                    RetrievalCandidate(
                        chunk=chunk,
                        score=hit.score,
                        source="qdrant"
                    )
                )
            return candidates

        except Exception as e:
            self.had_fallback = True
            self.fallback_reason = f"Vector search query failed: {str(e)}"
            logger.error(f"Qdrant vector search failed: {e}. Falling back to DB search.")
            if self.fallback_searcher:
                return self.fallback_searcher.search(routed_question, limit)
            raise


class HydeVectorSearcher(VectorSearchPort):
    """Vector searcher wrapper that applies Hypothetical Document Embeddings (HyDE) before search."""

    def __init__(self, vector_searcher: VectorSearchPort, generator: Any):
        self.vector_searcher = vector_searcher
        self.generator = generator
        self.last_hyde_doc = None

    def search(self, routed_question: RoutedQuestion, limit: int) -> Sequence[RetrievalCandidate]:
        # 1. Generate hypothetical document
        try:
            hyde_doc = self.generator.generate_hypothetical_document(routed_question.question)
        except Exception as e:
            logger.warning(f"HyDE generation failed with exception: {e}. Falling back to original question.")
            hyde_doc = routed_question.question

        self.last_hyde_doc = hyde_doc
        logger.info(f"HyDE generated hypothetical document: {hyde_doc}")

        # 2. Create a new RoutedQuestion with the hypothetical document as the question text
        hyde_routed_question = RoutedQuestion(
            question=hyde_doc,
            workspace_id=routed_question.workspace_id,
            branch_path=routed_question.branch_path,
            confidence=routed_question.confidence
        )

        # 3. Perform vector search using the hypothetical document
        return self.vector_searcher.search(hyde_routed_question, limit=limit)

