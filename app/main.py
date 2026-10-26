import logging
import os
import shutil
import sys
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import settings
from app.ingest import ingest_pdf
from app.llm import generate_answer
from app.models import (
    DeleteDocumentResponse,
    DocumentListResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.retrieval import _query_cache, retrieve
from app.vectorstore import (
    delete_document_by_name,
    get_vectorstore,
    list_ingested_documents,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag.api")

app = FastAPI(
    title="RAG Pipeline API",
    description="Production-ready LLM-powered Retrieval-Augmented Generation API with local embeddings and query cache.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"


@app.get("/", include_in_schema=False)
async def root():
    """Serve web studio frontend."""
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return {
        "title": "RAG Pipeline Studio",
        "status": "online",
        "documentation": "/docs",
    }


@app.get("/api", tags=["General"])
async def api_info():
    """API overview and endpoints list."""
    return {
        "title": "RAG Pipeline API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "ingest": "POST /ingest",
            "query": "POST /query",
            "list_documents": "GET /documents",
            "delete_document": "DELETE /documents/{filename}",
            "health": "GET /health",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health():
    try:
        vs = get_vectorstore()
        total_chunks = vs._collection.count()
    except Exception:
        total_chunks = 0

    return HealthResponse(
        status="healthy",
        vectorstore_chunks=total_chunks,
        cache_size=len(_query_cache),
        cache_capacity=_query_cache.capacity,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        llm_model=settings.LLM_MODEL,
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a PDF document into the vector store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    safe_filename = Path(file.filename).name
    save_path = os.path.join(settings.UPLOAD_DIR, safe_filename)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not save file: {str(e)}",
        )

    try:
        chunks_created, total_pages = ingest_pdf(save_path, safe_filename)
        return IngestResponse(
            filename=safe_filename,
            chunks_created=chunks_created,
            total_pages=total_pages,
            message=f"Successfully ingested {chunks_created} chunks across {total_pages} pages.",
        )
    except Exception as e:
        logger.error(f"Error processing PDF {safe_filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process and index PDF: {str(e)}",
        )


@app.post("/query", response_model=QueryResponse, tags=["Retrieval & Generation"])
async def query_pipeline(request: QueryRequest):
    """Query the RAG pipeline with grounded context retrieval and source citations."""
    try:
        # Step 1: Context retrieval with query caching
        chunks, retrieval_ms, cache_hit = retrieve(
            question=request.question, top_k=request.top_k
        )

        # Step 2: Strict grounded LLM response generation
        answer, generation_ms = generate_answer(
            question=request.question, context_chunks=chunks
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=chunks,
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            cache_hit=cache_hit,
        )
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query error: {str(e)}",
        )


@app.get("/documents", response_model=DocumentListResponse, tags=["Document Management"])
async def list_documents():
    """List all ingested documents and their chunk counts."""
    docs = list_ingested_documents()
    total_chunks = sum(d["chunk_count"] for d in docs)
    return DocumentListResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=total_chunks,
    )


@app.delete(
    "/documents/{filename}",
    response_model=DeleteDocumentResponse,
    tags=["Document Management"],
)
async def delete_document(filename: str):
    """Delete all chunks for a specific document from the vector store."""
    try:
        deleted_count = delete_document_by_name(filename)
        if deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{filename}' not found in vector store.",
            )
        return DeleteDocumentResponse(
            filename=filename,
            deleted_chunks=deleted_count,
            message=f"Successfully deleted {deleted_count} chunks for '{filename}'.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Delete failed: {str(e)}",
        )
