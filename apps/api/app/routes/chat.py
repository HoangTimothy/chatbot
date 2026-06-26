import logging
from datetime import datetime, timezone
from typing import Annotated, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.db.session import get_db_session, SessionLocal
from app.routes.auth import get_current_user
from app.routes.workspaces import get_current_workspace
from app.config import settings

from shared.models import User, Workspace, Conversation, Message, RetrievalTrace, Chunk as DbChunk
from rag_core.services.router import DomainRouter
from rag_core.adapters.searchers import SQLDatabaseSearcher, ElasticsearchSearcher, QdrantSearcher
from rag_core.adapters.reranker import RerankerAdapter
from rag_core.services.context_selector import TokenAwareContextSelector
from rag_core.adapters.generator import OpenAIGenerator
from rag_core.flows.retrieval_flow import RetrievalPipeline
from rag_core.flows.chat_flow import ChatPipeline

logger = logging.getLogger("api.chat")
router = APIRouter(prefix="/chat", tags=["chat"])


# --- Pydantic Schema Models ---

class SessionCreateRequest(BaseModel):
    title: str | None = None


class SessionResponse(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class MessageCreateRequest(BaseModel):
    content: str
    enable_hyde: bool | None = None


class GroundedAnswerResponse(BaseModel):
    answer: str
    citations: List[str]
    insufficient_context: bool
    message_id: str


class RetrievalTraceResponse(BaseModel):
    id: str
    message_id: str
    routed_branch: str | None
    hybrid_results: Any
    reranked_results: Any
    query_tokens: int | None
    response_tokens: int | None = None
    context_tokens: int | None = None
    total_tokens: int | None = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- FastAPI Endpoint Definitions ---

@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_session(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)],
    request: SessionCreateRequest = SessionCreateRequest()
):
    """Create a new chat conversation session in the current workspace context."""
    title = request.title or "New Chat"
    session = Conversation(
        workspace_id=workspace.id,
        user_id=current_user.id,
        title=title
    )
    db.add(session)
    await db.flush()
    await db.commit()
    await db.refresh(session)
    return session


@router.get("/sessions", response_model=List[SessionResponse])
async def list_chat_sessions(
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Retrieve all conversations in the workspace belonging to the user."""
    stmt = (
        select(Conversation)
        .where(
            Conversation.workspace_id == workspace.id,
            Conversation.user_id == current_user.id
        )
        .order_by(Conversation.updated_at.desc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()


class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
async def list_session_messages(
    session_id: str,
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Retrieve all message logs inside a specific conversation session."""
    session_stmt = select(Conversation).where(
        Conversation.id == session_id,
        Conversation.workspace_id == workspace.id,
        Conversation.user_id == current_user.id
    )
    session_result = await db.execute(session_stmt)
    if not session_result.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found."
        )

    stmt = (
        select(Message)
        .where(Message.conversation_id == session_id)
        .order_by(Message.created_at.asc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()


def check_searcher_fallback(searcher) -> bool:
    if searcher is None:
        return False
    if getattr(searcher, "had_fallback", False):
        return True
    if hasattr(searcher, "vector_searcher"):
        return check_searcher_fallback(searcher.vector_searcher)
    return False


def get_searcher_fallback_reason(searcher) -> str | None:
    if searcher is None:
        return None
    if getattr(searcher, "had_fallback", False):
        return getattr(searcher, "fallback_reason", "Fallback triggered")
    if hasattr(searcher, "vector_searcher"):
        return get_searcher_fallback_reason(searcher.vector_searcher)
    return None


@router.post("/sessions/{session_id}/messages", response_model=GroundedAnswerResponse)
async def post_chat_message(
    session_id: str,
    request: MessageCreateRequest,
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Append a new user question, run RAG retrieval + generation, save trace & response."""
    
    # 1. Retrieve the conversation context
    session_stmt = select(Conversation).where(
        Conversation.id == session_id,
        Conversation.workspace_id == workspace.id,
        Conversation.user_id == current_user.id
    )
    session_result = await db.execute(session_stmt)
    conversation = session_result.scalars().first()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found in this workspace."
        )

    # 2. Extract conversation history (last 10 messages) before adding new user message
    history_stmt = (
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.desc())
        .limit(10)
    )
    history_result = await db.execute(history_stmt)
    history_messages = list(history_result.scalars().all())
    history_messages.reverse()  # chronological order
    
    chat_history = [
        {"role": msg.role, "content": msg.content}
        for msg in history_messages
    ]

    # 3. Dynamic branch routing lookup
    branches_stmt = (
        select(DbChunk.knowledge_branch_path)
        .where(DbChunk.workspace_id == workspace.id)
        .distinct()
    )
    branches_result = await db.execute(branches_stmt)
    raw_branches = branches_result.scalars().all()
    
    available_branches = []
    for path in raw_branches:
        if path:
            available_branches.append(tuple(path.split("/")))

    # 4. Construct RAG components
    sql_fallback = SQLDatabaseSearcher(db_session_factory=SessionLocal)
    
    keyword_searcher = ElasticsearchSearcher(
        es_url=settings.ELASTICSEARCH_URL,
        fallback_searcher=sql_fallback
    )
    
    generator = OpenAIGenerator(
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.LLM_MODEL
    )
    
    vector_searcher = QdrantSearcher(
        qdrant_url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION,
        embedding_model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY,
        google_api_key=settings.GOOGLE_API_KEY,
        fallback_searcher=sql_fallback
    )
    
    # Enable HyDE wrapper if requested/configured
    use_hyde = request.enable_hyde if request.enable_hyde is not None else settings.ENABLE_HYDE
    if use_hyde:
        from rag_core.adapters.searchers import HydeVectorSearcher
        vector_searcher = HydeVectorSearcher(
            vector_searcher=vector_searcher,
            generator=generator
        )
    
    reranker = RerankerAdapter(
        provider=settings.RERANKER_PROVIDER,
        model_name=settings.RERANKER_MODEL,
        api_key=settings.OPENAI_API_KEY
    )
    
    context_selector = TokenAwareContextSelector(max_tokens=4000)

    # 5. Build Pipeline — choose multi-source or legacy based on config
    router_instance = DomainRouter(
        available_branches=available_branches,
        openai_api_key=settings.OPENAI_API_KEY,
        model_name=settings.LLM_MODEL
    )

    retrieval_pipeline = RetrievalPipeline(
        router=router_instance,
        keyword_searcher=keyword_searcher,
        vector_searcher=vector_searcher,
        reranker=reranker
    )

    # Determine if multi-source routing is enabled
    use_multi_source = settings.ENABLE_QUERY_ROUTING and (
        settings.ENABLE_WEB_SEARCH or settings.ENABLE_KNOWLEDGE_GRAPH
    )

    multi_source_pipeline = None
    query_route = None
    kg_result = None

    if use_multi_source:
        from rag_core.services.query_router import QueryRouter
        from rag_core.flows.multi_source_flow import MultiSourceRetrievalPipeline

        # Construct QueryRouter
        query_router = QueryRouter(
            available_branches=available_branches,
            openai_api_key=settings.OPENAI_API_KEY,
            google_api_key=settings.GOOGLE_API_KEY,
            model_name=settings.LLM_MODEL,
        )

        # Optional: Web search adapter
        web_searcher = None
        if settings.ENABLE_WEB_SEARCH and settings.TAVILY_API_KEY:
            from rag_core.adapters.web_searcher import TavilyWebSearcher
            web_searcher = TavilyWebSearcher(
                api_key=settings.TAVILY_API_KEY,
            )

        # Optional: Knowledge graph adapter
        knowledge_graph = None
        if settings.ENABLE_KNOWLEDGE_GRAPH:
            from rag_core.adapters.knowledge_graph import NetworkXKnowledgeGraph
            knowledge_graph = NetworkXKnowledgeGraph(
                persist_path=settings.KG_PERSIST_PATH,
            )

        multi_source_pipeline = MultiSourceRetrievalPipeline(
            query_router=query_router,
            kb_pipeline=retrieval_pipeline,
            reranker=reranker,
            web_searcher=web_searcher,
            knowledge_graph=knowledge_graph,
            web_search_enabled=settings.ENABLE_WEB_SEARCH,
            kg_available=settings.ENABLE_KNOWLEDGE_GRAPH and knowledge_graph is not None,
        )

    chat_pipeline = ChatPipeline(
        retrieval_pipeline=retrieval_pipeline,
        context_selector=context_selector,
        generator=generator,
        multi_source_pipeline=multi_source_pipeline,
    )

    # 6. Execute pipeline
    if use_multi_source:
        query_route, candidates, selected_context, grounded_answer, kg_result = (
            chat_pipeline.generate_response_multi_source(
                workspace_id=workspace.id,
                question=request.content,
                available_branches=available_branches,
                chat_history=chat_history,
            )
        )
        routed_branch_str = "/".join(query_route.branch_path) if query_route.branch_path else "ROOT"
    else:
        routed_question, candidates, selected_context, grounded_answer = chat_pipeline.generate_response(
            workspace_id=workspace.id,
            question=request.content,
            chat_history=chat_history
        )
        routed_branch_str = "/".join(routed_question.branch_path) if routed_question.branch_path else "ROOT"

    # 7. Persist conversation states & traces to Database
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.content
    )
    db.add(user_msg)
    await db.flush()

    assistant_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=grounded_answer.answer
    )
    db.add(assistant_msg)
    await db.flush()

    # Update conversation last updated timestamp
    conversation.updated_at = datetime.now(timezone.utc)

    # Token counting using tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        query_tokens = len(enc.encode(request.content))
        response_tokens = len(enc.encode(grounded_answer.answer))
    except Exception:
        query_tokens = len(request.content.split())
        response_tokens = len(grounded_answer.answer.split())
        
    context_tokens = selected_context.total_tokens if selected_context else 0
    total_tokens = query_tokens + response_tokens + context_tokens

    # Format trace results
    reranked_serialized = [
        {
            "chunk_id": c.chunk.chunk_id,
            "text": c.chunk.text,
            "score": c.score,
            "source": c.source,
            "knowledge_branch_path": "/".join(c.chunk.knowledge_branch_path) if c.chunk.knowledge_branch_path else ""
        }
        for c in candidates
    ]

    # Check if HyDE was used and retrieve the hypothetical document
    hyde_doc = getattr(vector_searcher, "last_hyde_doc", None)
    
    keyword_fallback = check_searcher_fallback(keyword_searcher)
    keyword_reason = get_searcher_fallback_reason(keyword_searcher)
    
    vector_fallback = check_searcher_fallback(vector_searcher)
    vector_reason = get_searcher_fallback_reason(vector_searcher)
    
    generation_fallback = getattr(generator, "had_fallback", False)
    generation_reason = getattr(generator, "fallback_reason", None)

    # Build extended trace with multi-source routing info
    query_routing_info = None
    if query_route:
        query_routing_info = {
            "strategy": query_route.strategy.value,
            "confidence": query_route.confidence,
            "reasoning": query_route.reasoning,
            "sub_queries": list(query_route.sub_queries),
        }

    kg_result_info = None
    if kg_result:
        kg_result_info = {
            "entity_count": len(kg_result.entities),
            "relation_count": len(kg_result.relations),
            "natural_language_answer": kg_result.natural_language_answer[:500],
        }
    
    hybrid_results_serialized = {
        "candidates": reranked_serialized,
        "hyde_document": hyde_doc,
        "query_routing": query_routing_info,
        "knowledge_graph": kg_result_info,
        "fallbacks": {
            "keyword_search": keyword_fallback,
            "vector_search": vector_fallback,
            "answer_generation": generation_fallback
        },
        "fallback_reasons": {
            "keyword_search": keyword_reason,
            "vector_search": vector_reason,
            "answer_generation": generation_reason
        },
        "token_metrics": {
            "query_tokens": query_tokens,
            "response_tokens": response_tokens,
            "context_tokens": context_tokens,
            "total_tokens": total_tokens
        }
    }

    trace = RetrievalTrace(
        message_id=assistant_msg.id,
        routed_branch=routed_branch_str,
        hybrid_results=hybrid_results_serialized,
        reranked_results=reranked_serialized,
        query_tokens=query_tokens
    )
    db.add(trace)

    await db.commit()

    return GroundedAnswerResponse(
        answer=grounded_answer.answer,
        citations=list(grounded_answer.citations),
        insufficient_context=grounded_answer.insufficient_context,
        message_id=assistant_msg.id
    )


@router.get("/sessions/{session_id}/trace/{message_id}", response_model=RetrievalTraceResponse)
async def get_message_retrieval_trace(
    session_id: str,
    message_id: str,
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Retrieve RAG pipeline trace metadata matching a message context."""
    session_stmt = select(Conversation).where(
        Conversation.id == session_id,
        Conversation.workspace_id == workspace.id,
        Conversation.user_id == current_user.id
    )
    session_result = await db.execute(session_stmt)
    conversation = session_result.scalars().first()
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found."
        )

    trace_stmt = select(RetrievalTrace).where(RetrievalTrace.message_id == message_id)
    trace_result = await db.execute(trace_stmt)
    trace = trace_result.scalars().first()
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Retrieval trace not found for this message."
        )

    # Resolve token metrics
    response_tokens = None
    context_tokens = None
    total_tokens = None
    
    if trace.hybrid_results and isinstance(trace.hybrid_results, dict):
        token_metrics = trace.hybrid_results.get("token_metrics")
        if token_metrics:
            response_tokens = token_metrics.get("response_tokens")
            context_tokens = token_metrics.get("context_tokens")
            total_tokens = token_metrics.get("total_tokens")

    if response_tokens is None:
        # Fallback counting on message content
        msg_stmt = select(Message).where(Message.id == message_id)
        msg_result = await db.execute(msg_stmt)
        msg = msg_result.scalars().first()
        if msg:
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                response_tokens = len(enc.encode(msg.content))
            except Exception:
                response_tokens = len(msg.content.split())
        else:
            response_tokens = 0
            
    if context_tokens is None:
        ctx_text = "".join([c.get("text", "") for c in trace.reranked_results]) if trace.reranked_results else ""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            context_tokens = len(enc.encode(ctx_text))
        except Exception:
            context_tokens = len(ctx_text.split())

    if total_tokens is None:
        q_tok = trace.query_tokens or 0
        r_tok = response_tokens or 0
        c_tok = context_tokens or 0
        total_tokens = q_tok + r_tok + c_tok

    return RetrievalTraceResponse(
        id=trace.id,
        message_id=trace.message_id,
        routed_branch=trace.routed_branch,
        hybrid_results=trace.hybrid_results,
        reranked_results=trace.reranked_results,
        query_tokens=trace.query_tokens,
        response_tokens=response_tokens,
        context_tokens=context_tokens,
        total_tokens=total_tokens,
        created_at=trace.created_at
    )


class FeedbackCreateRequest(BaseModel):
    rating: str  # "upvote" or "downvote"
    comment: str | None = None


@router.post("/sessions/{session_id}/messages/{message_id}/feedback", status_code=status.HTTP_201_CREATED)
async def submit_message_feedback(
    session_id: str,
    message_id: str,
    request: FeedbackCreateRequest,
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Submit rating (upvote/downvote) feedback for a generated response message."""
    session_stmt = select(Conversation).where(
        Conversation.id == session_id,
        Conversation.workspace_id == workspace.id,
        Conversation.user_id == current_user.id
    )
    session_result = await db.execute(session_stmt)
    if not session_result.scalars().first():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found."
        )

    message_stmt = select(Message).where(
        Message.id == message_id,
        Message.conversation_id == session_id
    )
    message_result = await db.execute(message_stmt)
    message = message_result.scalars().first()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found."
        )

    from shared.models import AnswerFeedback
    fb_stmt = select(AnswerFeedback).where(
        AnswerFeedback.message_id == message_id,
        AnswerFeedback.user_id == current_user.id
    )
    fb_result = await db.execute(fb_stmt)
    existing_fb = fb_result.scalars().first()

    if existing_fb:
        existing_fb.rating = request.rating
        existing_fb.comment = request.comment
        await db.commit()
        return {"detail": "Feedback updated successfully."}

    feedback = AnswerFeedback(
        message_id=message_id,
        user_id=current_user.id,
        rating=request.rating,
        comment=request.comment
    )
    db.add(feedback)
    await db.commit()
    return {"detail": "Feedback submitted successfully."}


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    workspace: Annotated[Workspace, Depends(get_current_workspace)],
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db_session)]
):
    """Permanently delete a chat session and all its messages."""
    stmt = select(Conversation).where(
        Conversation.id == session_id,
        Conversation.workspace_id == workspace.id,
        Conversation.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalars().first()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found."
        )

    await db.delete(session)
    await db.commit()
    return {"detail": "Chat session deleted successfully."}

