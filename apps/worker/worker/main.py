import os
import sys
import time
import logging
import pathlib

# Add project paths to sys.path to allow imports from packages/shared, packages/rag_core, and apps/api
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "packages", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "packages", "rag_core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "apps", "api")))

from app.adapters.db.session import SessionLocal
from app.config import settings
from shared.enums import DocumentStatus, IngestionJobStatus
from shared.models import Document, DocumentVersion, IngestionJob, Chunk as DbChunk

from rag_core.adapters.parsers import ParserRegistry
from rag_core.services.chunking import SemanticChunker
from rag_core.adapters.indexers import ElasticsearchIndexer, QdrantIndexer
from rag_core.contracts.types import DocumentRef

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("rag_worker")


def process_job(job_id: str) -> None:
    """Process a single queued ingestion job."""
    db = SessionLocal()
    try:
        # Fetch job, document, and current version in a single session
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if not job or job.status != IngestionJobStatus.RUNNING:
            return

        document = db.query(Document).filter(Document.id == job.document_id).first()
        if not document:
            raise ValueError(f"Document {job.document_id} not found in database.")

        version = db.query(DocumentVersion).filter(DocumentVersion.id == document.current_version_id).first()
        if not version:
            raise ValueError(f"DocumentVersion {document.current_version_id} not found in database.")

        logger.info(f"Starting ingestion processing for document: '{document.name}' (ID: {document.id})")

        # Get full local path to file in mock ObjectStorage
        storage_base = pathlib.Path(settings.OBJECT_STORAGE_LOCAL_PATH).resolve()
        full_file_path = storage_base / document.file_path

        if not full_file_path.exists():
            raise FileNotFoundError(f"File not found in local object storage: {full_file_path}")

        # 1. Parse File
        file_ext = pathlib.Path(document.name).suffix
        logger.info(f"Parsing file with extension: '{file_ext}'")
        parser = ParserRegistry().get_parser(file_ext)
        blocks = parser.parse(str(full_file_path))
        
        if not blocks:
            raise ValueError("Parser returned empty text blocks. File may be empty or failed text extraction.")

        # 2. Semantic Chunking
        doc_ref = DocumentRef(
            workspace_id=document.workspace_id,
            document_id=document.id,
            document_version_id=version.id,
            file_name=document.name,
            file_hash=version.file_hash
        )
        logger.info("Executing semantic chunking and feature extraction...")
        chunker = SemanticChunker(document_ref=doc_ref)
        # Default knowledge branch path is general
        branch_path = ("general",)
        chunks = chunker.chunk(blocks, branch_path=branch_path)

        # 2.5 Apply Contextual Retrieval query situated chunk text if enabled
        if getattr(settings, "ENABLE_CONTEXTUAL_RETRIEVAL", False):
            logger.info("Contextual Retrieval is enabled. Situating chunk text within document context...")
            from rag_core.adapters.generator import OpenAIGenerator
            from rag_core.contracts.types import Chunk
            
            generator = OpenAIGenerator(
                openai_api_key=settings.OPENAI_API_KEY,
                model_name=settings.LLM_MODEL
            )
            
            document_text = "\n\n".join([b.text for b in blocks])
            
            contextualized_chunks = []
            for c in chunks:
                prefix = generator.generate_contextual_prefix(
                    document_text=document_text,
                    chunk_text=c.text
                )
                if prefix:
                    new_text = f"[Context: {prefix}]\n\n{c.text}"
                    encoder = getattr(chunker, "encoder", None)
                    if encoder:
                        new_tokens = len(encoder.encode(new_text))
                    else:
                        new_tokens = len(new_text.split())
                    
                    new_features = dict(c.features)
                    new_features["char_count"] = len(new_text)
                    new_features["contextual_prefix"] = prefix
                    
                    c = Chunk(
                        chunk_id=c.chunk_id,
                        document=c.document,
                        text=new_text,
                        token_count=new_tokens,
                        knowledge_branch_path=c.knowledge_branch_path,
                        features=new_features
                    )
                contextualized_chunks.append(c)
            chunks = contextualized_chunks

        if not chunks:
            raise ValueError("Chunker failed to generate any chunks from parsed text.")

        # 2.9 Clean up old chunks for this document across all backends to prevent stale data pollution
        logger.info(f"Clearing old chunks for document {document.id} across SQLite, Elasticsearch, and Qdrant...")
        db.query(DbChunk).filter(DbChunk.document_id == document.id).delete()
        
        try:
            es_indexer = ElasticsearchIndexer(
                es_url=settings.ELASTICSEARCH_URL,
                index_name=settings.QDRANT_COLLECTION
            )
            if es_indexer._is_enabled and es_indexer.client:
                es_indexer.client.delete_by_query(
                    index=es_indexer.index_name,
                    body={"query": {"term": {"document_id": document.id}}}
                )
        except Exception as es_ex:
            logger.warning(f"Failed to clear ES index before update: {es_ex}")

        try:
            qdrant_indexer = QdrantIndexer(
                qdrant_url=settings.QDRANT_URL,
                collection_name=settings.QDRANT_COLLECTION,
                embedding_model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                google_api_key=settings.GOOGLE_API_KEY,
            )
            if qdrant_indexer._is_enabled and qdrant_indexer.client:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                qdrant_indexer.client.delete(
                    collection_name=qdrant_indexer.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document.id)
                            )
                        ]
                    )
                )
        except Exception as qd_ex:
            logger.warning(f"Failed to clear Qdrant collection before update: {qd_ex}")

        # 3. Persist Chunks in SQL Database
        logger.info(f"Persisting {len(chunks)} chunks in SQL Database...")
        for chunk in chunks:
            db_chunk = DbChunk(
                id=chunk.chunk_id,
                workspace_id=document.workspace_id,
                document_id=document.id,
                document_version_id=version.id,
                source_file_name=chunk.document.file_name,
                source_file_hash=chunk.document.file_hash,
                page_number=chunk.features.get("page_number"),
                sheet_name=chunk.features.get("sheet_name"),
                section_title=chunk.features.get("section_title"),
                heading_path=chunk.features.get("heading_path"),
                knowledge_branch_path="/".join(chunk.knowledge_branch_path),
                text=chunk.text,
                token_count=chunk.token_count,
                char_count=chunk.features.get("char_count", len(chunk.text)),
                table_count=chunk.features.get("table_count", 0),
                image_count=chunk.features.get("image_count", 0),
                contains_policy_language=chunk.features.get("contains_policy_language", False),
                contains_product_spec=chunk.features.get("contains_product_spec", False),
                contains_procedure_steps=chunk.features.get("contains_procedure_steps", False),
                chunk_quality_score=chunk.features.get("chunk_quality_score", 1.0),
                embedding_model=settings.EMBEDDING_MODEL,
                chunking_strategy=chunk.features.get("chunking_strategy", "semantic_heading"),
                chunk_version=chunk.features.get("chunk_version", 1)
            )
            db.add(db_chunk)

        # 4. Index in Keyword Store (Elasticsearch)
        logger.info("Indexing chunks in Elasticsearch...")
        es_indexer = ElasticsearchIndexer(
            es_url=settings.ELASTICSEARCH_URL,
            index_name=settings.QDRANT_COLLECTION
        )
        es_indexer.index_chunks(chunks)

        # 5. Index in Vector Store (Qdrant)
        logger.info("Indexing chunks in Qdrant Vector DB...")
        qdrant_indexer = QdrantIndexer(
            qdrant_url=settings.QDRANT_URL,
            collection_name=settings.QDRANT_COLLECTION,
            embedding_model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        qdrant_indexer.index_chunks(chunks)

        # 6. Extract and build Knowledge Graph if enabled
        if getattr(settings, "ENABLE_KNOWLEDGE_GRAPH", False) and getattr(settings, "KG_AUTO_EXTRACT", False):
            logger.info("Knowledge Graph Auto-Extraction is enabled. Extracting entities and relations...")
            try:
                from rag_core.adapters.knowledge_graph import NetworkXKnowledgeGraph
                from rag_core.services.kg_extractor import KGExtractor

                kg = NetworkXKnowledgeGraph(persist_path=settings.KG_PERSIST_PATH)
                extractor = KGExtractor(
                    openai_api_key=settings.OPENAI_API_KEY,
                    google_api_key=settings.GOOGLE_API_KEY,
                    model_name=settings.LLM_MODEL
                )
                
                # Extract entities/relations chunk by chunk
                for idx, c in enumerate(chunks, start=1):
                    logger.info(f"Extracting entities from chunk {idx}/{len(chunks)}...")
                    entities, relations = extractor.extract(c.text, doc_ref)
                    kg.add_entities(entities)
                    kg.add_relations(relations)
                
                kg.persist(settings.KG_PERSIST_PATH)
                logger.info(f"Knowledge graph persisted successfully. Nodes: {kg.node_count}, Edges: {kg.edge_count}")
            except Exception as kg_ex:
                logger.error(f"Failed to perform knowledge graph extraction: {kg_ex}")

        # Update job and document status to ready
        job.status = IngestionJobStatus.COMPLETED
        document.status = DocumentStatus.READY
        db.commit()
        logger.info(f"Ingestion job completed successfully for document: '{document.name}'")

    except Exception as e:
        db.rollback()
        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
        # Update statuses to failed
        try:
            err_db = SessionLocal()
            job = err_db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
            if job:
                job.status = IngestionJobStatus.FAILED
                job.error_message = str(e)
                document = err_db.query(Document).filter(Document.id == job.document_id).first()
                if document:
                    document.status = DocumentStatus.FAILED
                err_db.commit()
            err_db.close()
        except Exception as inner_ex:
            logger.error(f"Failed to record job error status in DB: {inner_ex}")

    finally:
        db.close()


def poll_and_execute() -> None:
    """Run polling loop querying database for QUEUED tasks."""
    logger.info("Ingestion Worker started. Polling database for QUEUED jobs...")
    
    while True:
        db = SessionLocal()
        job = None
        try:
            # Query first queued job
            job = db.query(IngestionJob).filter(IngestionJob.status == IngestionJobStatus.QUEUED).first()
            if job:
                # Atomically claim job
                job.status = IngestionJobStatus.RUNNING
                job.document.status = DocumentStatus.PROCESSING
                db.commit()
                job_id = job.id
            else:
                job_id = None
        except Exception as e:
            logger.error(f"Database polling error: {e}")
            job_id = None
        finally:
            db.close()

        if job_id:
            process_job(job_id)
        else:
            time.sleep(2)


if __name__ == "__main__":
    try:
        poll_and_execute()
    except KeyboardInterrupt:
        logger.info("Ingestion Worker shut down by user.")
        sys.exit(0)
