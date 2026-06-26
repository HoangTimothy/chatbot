import os
import random
import logging
from typing import Sequence
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from rag_core.contracts.types import Chunk

logger = logging.getLogger("rag_core.indexers")


class ElasticsearchIndexer:
    """Indexer for keyword search using Elasticsearch/OpenSearch."""

    def __init__(self, es_url: str, index_name: str = "enterprise_chunks"):
        self.es_url = es_url
        self.index_name = index_name
        self.client = None
        self._is_enabled = False

        try:
            self.client = Elasticsearch(self.es_url)
            # Try to ping the service
            if self.client.ping():
                self._is_enabled = True
                self._ensure_index()
            else:
                logger.warning(f"Elasticsearch ping failed at: {es_url}. Falling back to mock indexing.")
        except Exception as e:
            logger.warning(f"Failed to connect to Elasticsearch: {e}. Falling back to mock indexing.")

    def _ensure_index(self) -> None:
        """Create the target index with correct mapping if missing."""
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "workspace_id": {"type": "keyword"},
                        "document_id": {"type": "keyword"},
                        "document_version_id": {"type": "keyword"},
                        "source_file_name": {"type": "text"},
                        "text": {"type": "text", "analyzer": "standard"},
                        "knowledge_branch_path": {"type": "keyword"},
                        "created_at": {"type": "date"},
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=mapping)
            logger.info(f"Created Elasticsearch index: '{self.index_name}'")

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks into Elasticsearch."""
        if not self._is_enabled or not self.client:
            logger.warning("Elasticsearch is disabled or offline. Skipping keyword index upload.")
            return

        for chunk in chunks:
            doc_body = {
                "chunk_id": chunk.chunk_id,
                "workspace_id": chunk.document.workspace_id,
                "document_id": chunk.document.document_id,
                "document_version_id": chunk.document.document_version_id,
                "source_file_name": chunk.document.file_name,
                "text": chunk.text,
                "knowledge_branch_path": list(chunk.knowledge_branch_path),
            }
            self.client.index(index=self.index_name, id=chunk.chunk_id, body=doc_body)
        logger.info(f"Successfully indexed {len(chunks)} chunks in Elasticsearch.")


class QdrantIndexer:
    """Indexer for semantic vector search using Qdrant.

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
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.embedding_model = embedding_model
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

        # Initialise provider-specific clients
        self.openai_client = None
        if self.provider != "google":
            if openai_api_key:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_api_key)
            else:
                logger.warning("No OpenAI API key provided. Embeddings will fall back to mock values.")

        if self.provider == "google" and google_api_key:
            os.environ.setdefault("GOOGLE_API_KEY", google_api_key)

        try:
            self.client = QdrantClient(url=self.qdrant_url)
            # Run simple query to check connection
            self.client.get_collections()
            self._is_enabled = True
            self._ensure_collection()
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant at: {qdrant_url} ({e}). Falling back to mock indexing.")

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if not existing."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: '{self.collection_name}' with size {self.dimension}")

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
                    task_type="retrieval_document",
                    request_options={"timeout": 30.0}
                )
                return response["embedding"]
            else:
                # OpenAI path
                if not self.openai_client:
                    random.seed(hash(text))
                    return [random.uniform(-1.0, 1.0) for _ in range(self.dimension)]

                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=text,
                )
                return response.data[0].embedding

        except Exception as e:
            logger.warning(f"Embedding generation failed ({self.provider}): {e}. Falling back to mock vector.")
            random.seed(hash(text))
            return [random.uniform(-1.0, 1.0) for _ in range(self.dimension)]

    def index_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Index chunks into Qdrant vector database."""
        if not self._is_enabled or not self.client:
            logger.warning("Qdrant client is disabled or offline. Skipping vector upload.")
            return

        points = []
        for chunk in chunks:
            vector = self._get_embedding(chunk.text)

            # Construct payload metadata matching requirements
            payload = {
                "chunk_id": chunk.chunk_id,
                "workspace_id": chunk.document.workspace_id,
                "document_id": chunk.document.document_id,
                "document_version_id": chunk.document.document_version_id,
                "source_file_name": chunk.document.file_name,
                "text": chunk.text,
                "knowledge_branch_path": list(chunk.knowledge_branch_path),
            }
            # Flatten dataclass features dict into payload
            for k, v in chunk.features.items():
                payload[k] = v

            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points,
        )
        logger.info(f"Successfully uploaded {len(chunks)} chunks to Qdrant.")
