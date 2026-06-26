import os
import sys

# Add project paths to sys.path to allow imports from packages/shared, packages/rag_core, and apps/api
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "packages", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "packages", "rag_core")))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import auth, workspaces, documents, chat, admin


def create_app() -> FastAPI:
    """Create the FastAPI API app instance and register routing/middleware."""
    app = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        description="Enterprise RAG Platform API backend boundary.",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(auth.router)
    app.include_router(workspaces.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    app.include_router(admin.router)




    @app.get("/health", tags=["health"])
    async def health_check():
        """Basic service health check."""
        return {"status": "ok", "app_name": settings.APP_NAME}

    return app


app = create_app()


