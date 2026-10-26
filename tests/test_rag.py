import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval import QueryCache
from app.models import QueryRequest, QueryResponse, SourceChunk


client = TestClient(app)


def test_query_cache_operations():
    cache = QueryCache(capacity=2)
    assert len(cache) == 0

    cache.put("q1", [0.1, 0.2])
    assert cache.get("q1") == [0.1, 0.2]
    assert len(cache) == 1

    cache.put("q2", [0.3, 0.4])
    assert len(cache) == 2

    # Access q1 to make it most recently used
    _ = cache.get("q1")

    # Adding q3 should evict q2
    cache.put("q3", [0.5, 0.6])
    assert cache.get("q2") is None
    assert cache.get("q1") == [0.1, 0.2]
    assert cache.get("q3") == [0.5, 0.6]

    cache.clear()
    assert len(cache) == 0
    assert cache.get("q1") is None


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "RAG Pipeline" in response.text


def test_api_info():
    response = client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "cache_capacity" in data
    assert "vectorstore_chunks" in data


def test_ingest_invalid_file_type():
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", b"plain text content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF files are supported" in response.json()["detail"]


def test_models_validation():
    chunk = SourceChunk(file_name="sample.pdf", page=1, content="Sample text", score=0.95)
    assert chunk.file_name == "sample.pdf"
    assert chunk.page == 1

    req = QueryRequest(question="What is the policy?")
    assert req.question == "What is the policy?"

    resp = QueryResponse(
        question="What is the policy?",
        answer="Sample answer [Document: sample.pdf, Page: 1]",
        sources=[chunk],
        retrieval_ms=12.5,
        generation_ms=450.0,
        cache_hit=False
    )
    assert resp.cache_hit is False
    assert len(resp.sources) == 1
