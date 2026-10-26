import logging
from typing import Optional, List, Dict, Any
from app.config import settings

logger = logging.getLogger("rag.vectorstore")

_embeddings: Optional[Any] = None
_vectorstore: Optional[Any] = None


def get_embeddings():
    """Singleton getter for HuggingFace embeddings model (lazy loaded)."""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info(
            f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME} on {settings.EMBEDDING_DEVICE}"
        )
        model_kwargs = {"device": settings.EMBEDDING_DEVICE}
        encode_kwargs = {"normalize_embeddings": True}
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    return _embeddings


def get_vectorstore():
    """Singleton getter for Chroma vector store (lazy loaded)."""
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma

        logger.info(
            f"Initializing Chroma vector store at {settings.CHROMA_PERSIST_DIRECTORY}"
        )
        _vectorstore = Chroma(
            collection_name=settings.COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=settings.CHROMA_PERSIST_DIRECTORY,
        )
    return _vectorstore


def get_chroma_client():
    """Get persistent ChromaDB client directly without embedding model."""
    import chromadb
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIRECTORY)


def list_ingested_documents() -> List[Dict[str, Any]]:
    """List all ingested documents and their chunk counts without loading embeddings."""
    try:
        client = get_chroma_client()
        try:
            coll = client.get_collection(name=settings.COLLECTION_NAME)
        except Exception:
            return []

        data = coll.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        if not metadatas:
            return []

        doc_counts: Dict[str, int] = {}
        for meta in metadatas:
            if meta and "source" in meta:
                doc_counts[meta["source"]] = doc_counts.get(meta["source"], 0) + 1

        return [
            {"filename": fname, "chunk_count": count}
            for fname, count in sorted(doc_counts.items())
        ]
    except Exception as e:
        logger.error(f"Error querying documents from vectorstore: {e}")
        return []


def delete_document_by_name(filename: str) -> int:
    """Delete all chunks belonging to a specific document without loading embeddings."""
    try:
        client = get_chroma_client()
        try:
            coll = client.get_collection(name=settings.COLLECTION_NAME)
        except Exception:
            # Collection does not exist yet on a fresh server, nothing to delete
            return 0

        data = coll.get(where={"source": filename})
        ids = data.get("ids", [])
        if ids:
            coll.delete(ids=ids)
            return len(ids)
        return 0
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        return 0
