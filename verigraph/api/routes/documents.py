"""Document ingestion and management routes."""

from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from verigraph.api.schemas import DocumentIngestTextRequest, DocumentListResponse
from verigraph.config import settings
from verigraph.core.models import Chunk, Document
from verigraph.engine.pipeline import VeriGraphPipeline


router = APIRouter(prefix="/documents", tags=["Documents"])


def get_pipeline(request: Request) -> VeriGraphPipeline:
    return request.app.state.pipeline


@router.post("/upload", response_model=Document)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
) -> Document:
    """Uploads a PDF, Markdown, or text file and triggers the automated ingestion pipeline."""
    filename = file.filename or "uploaded_document.txt"
    dest_path = settings.uploads_dir / filename

    try:
        content = await file.read()
        dest_path.write_bytes(content)
        doc = pipeline.ingest_document(dest_path)
        return doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")


@router.post("/text", response_model=Document)
def ingest_raw_text(
    body: DocumentIngestTextRequest,
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
) -> Document:
    """Ingests raw markdown or plain text directly without a file upload."""
    slug = body.title.lower().replace(" ", "_")
    dest_path = settings.uploads_dir / f"{slug}.md"
    dest_path.write_text(body.content, encoding="utf-8")

    doc = pipeline.ingest_document(dest_path, title=body.title)
    return doc


@router.get("", response_model=DocumentListResponse)
def list_documents(
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
) -> DocumentListResponse:
    """Returns all currently indexed documents."""
    docs = pipeline.list_documents()
    return DocumentListResponse(documents=docs, total_count=len(docs))


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
):
    """Deletes a document and prunes its vector chunks and knowledge graph triples."""
    success = pipeline.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document successfully deleted", "document_id": document_id}


@router.get("/{document_id}/chunks", response_model=List[Chunk])
def get_document_chunks(
    document_id: str,
    pipeline: VeriGraphPipeline = Depends(get_pipeline),
) -> List[Chunk]:
    """Retrieves all chunks and section breadcrumbs for a specific document."""
    chunks = [
        c for c in pipeline.vector_store._chunks.values()
        if c.document_id == document_id
    ]
    chunks.sort(key=lambda x: x.chunk_index)
    return chunks
