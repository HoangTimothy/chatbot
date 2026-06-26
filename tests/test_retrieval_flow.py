import os
import sys

# Add project paths to sys.path to allow direct python execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "packages", "rag_core")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "api")))

import pytest

from app.adapters.db.session import SessionLocal
from rag_core.adapters.searchers import SQLDatabaseSearcher
from rag_core.adapters.reranker import RerankerAdapter
from rag_core.services.router import DomainRouter
from rag_core.flows.retrieval_flow import RetrievalPipeline
from shared.models import Workspace, Chunk as DbChunk, Document, DocumentVersion


def test_retrieval_pipeline_execution():
    """Verify routing, SQL fallback search, RRF merge, and reranking processes."""
    db = SessionLocal()
    try:
        # Resolve test workspace
        workspace = db.query(Workspace).filter(Workspace.name == "Default Workspace").first()
        assert workspace is not None

        # 1. Clean up existing chunks to start with clean state
        db.query(DbChunk).filter(DbChunk.workspace_id == workspace.id).delete()
        
        # Create a mock document and version to satisfy foreign key constraints
        doc = db.query(Document).filter(
            Document.workspace_id == workspace.id,
            Document.name == "mock_retrieval_doc.txt"
        ).first()
        if doc:
            db.delete(doc)
            db.commit()
            
        doc = Document(
            workspace_id=workspace.id,
            name="mock_retrieval_doc.txt",
            file_path="mock/path.txt",
            file_size=100,
            content_type="text/plain"
        )
        db.add(doc)
        db.flush()

        version = DocumentVersion(
            document_id=doc.id,
            version_number=1,
            file_hash="mockhash123",
            file_path="mock/path.txt"
        )
        db.add(version)
        db.flush()

        # Seed mock chunks into specific branches
        chunk1 = DbChunk(
            id="chunk-finance-01",
            workspace_id=workspace.id,
            document_id=doc.id,
            document_version_id=version.id,
            source_file_name=doc.name,
            source_file_hash=version.file_hash,
            knowledge_branch_path="finance/policy",
            text="All employees must submit expense reports by the 5th. Leave requests require direct manager approvals.",
            token_count=18,
            char_count=100
        )
        chunk2 = DbChunk(
            id="chunk-tech-02",
            workspace_id=workspace.id,
            document_id=doc.id,
            document_version_id=version.id,
            source_file_name=doc.name,
            source_file_hash=version.file_hash,
            knowledge_branch_path="technical/specs",
            text="The API server requires Python version 3.11 or greater and operates on port 6333 locally.",
            token_count=15,
            char_count=90
        )
        db.add_all([chunk1, chunk2])
        db.commit()

        # 2. Setup Retrieval Core components
        available_branches = [("finance", "policy"), ("technical", "specs")]
        router = DomainRouter(available_branches=available_branches)
        
        # Use SQL database searcher as fallback for both keyword and vector search
        sql_searcher = SQLDatabaseSearcher(db_session_factory=SessionLocal)
        reranker = RerankerAdapter(provider="local")
        
        pipeline = RetrievalPipeline(
            router=router,
            keyword_searcher=sql_searcher,
            vector_searcher=sql_searcher,
            reranker=reranker
        )

        # 3. Test query 1: Expect routing to technical/specs and matching chunk2
        q1 = "what are the server python version specifications?"
        routed_q1, candidates1 = pipeline.retrieve(workspace.id, q1)
        
        assert routed_q1.branch_path == ("technical", "specs")
        assert len(candidates1) == 1
        assert candidates1[0].chunk.chunk_id == "chunk-tech-02"
        assert "Python version" in candidates1[0].chunk.text

        # 4. Test query 2: Expect routing to finance/policy and matching chunk1
        q2 = "how do i submit my expense report policy?"
        routed_q2, candidates2 = pipeline.retrieve(workspace.id, q2)
        
        assert routed_q2.branch_path == ("finance", "policy")
        assert len(candidates2) == 1
        assert candidates2[0].chunk.chunk_id == "chunk-finance-01"
        assert "expense reports" in candidates2[0].chunk.text

    finally:
        db.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
