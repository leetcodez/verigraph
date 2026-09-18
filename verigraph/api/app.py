"""FastAPI Application Entrypoint with CORS and Static Frontend."""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from verigraph.config import settings
from verigraph.core.logging import logger
from verigraph.engine.pipeline import VeriGraphPipeline
from verigraph.api.routes.query import router as query_router
from verigraph.api.routes.documents import router as documents_router
from verigraph.api.routes.graph import router as graph_router
from verigraph.api.routes.benchmark import router as benchmark_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and tears down application resources."""
    logger.info("Initializing VeriGraph engine pipeline...")
    pipeline = VeriGraphPipeline()
    app.state.pipeline = pipeline

    # Ingest sample data on startup if database is empty
    sample_dir = settings.base_dir / "sample_data"
    if sample_dir.exists() and len(pipeline.list_documents()) == 0:
        logger.info("Empty database detected. Ingesting default reference datasets...")
        for sample_file in sample_dir.glob("*.md"):
            try:
                pipeline.ingest_document(sample_file)
            except Exception as e:
                logger.warning(f"Failed to ingest sample file {sample_file.name}: {e}")

    yield

    logger.info("Shutting down VeriGraph services.")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Enterprise Agentic GraphRAG & Verifiable Retrieval Engine",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routers under /api/v1
    api_prefix = "/api/v1"
    app.include_router(query_router, prefix=api_prefix)
    app.include_router(documents_router, prefix=api_prefix)
    app.include_router(graph_router, prefix=api_prefix)
    app.include_router(benchmark_router, prefix=api_prefix)

    @app.get("/health")
    def health_check():
        return {
            "status": "healthy",
            "version": settings.app_version,
            "documents_indexed": len(app.state.pipeline.list_documents()),
            "graph_nodes": app.state.pipeline.graph_engine.graph.number_of_nodes(),
            "graph_edges": app.state.pipeline.graph_engine.graph.number_of_edges(),
        }

    # Mount frontend static directory
    frontend_dir = settings.base_dir / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/")
        async def serve_index():
            index_path = frontend_dir / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return {"message": "VeriGraph API is active. Frontend index.html not found."}

    return app


app = create_app()
