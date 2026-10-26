from typing import List, Optional
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    file_name: str = Field(..., description="Name of the source file")
    page: int = Field(..., description="1-based page number")
    content: str = Field(..., description="Text content of the retrieved chunk")
    score: Optional[float] = Field(default=None, description="Similarity or relevance score")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question grounded in the ingested documents")
    top_k: Optional[int] = Field(default=None, ge=1, le=20, description="Override default number of chunks to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceChunk]
    retrieval_ms: float = Field(..., description="Retrieval latency in milliseconds")
    generation_ms: float = Field(..., description="LLM generation latency in milliseconds")
    cache_hit: bool = Field(..., description="Whether the query embedding was served from cache")


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    total_pages: int
    message: str


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total_documents: int
    total_chunks: int


class DeleteDocumentResponse(BaseModel):
    filename: str
    deleted_chunks: int
    message: str


class HealthResponse(BaseModel):
    status: str
    vectorstore_chunks: int
    cache_size: int
    cache_capacity: int
    embedding_model: str
    llm_model: str
