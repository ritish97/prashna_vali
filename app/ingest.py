import logging
from typing import Tuple
from app.config import settings
from app.vectorstore import get_vectorstore

logger = logging.getLogger("rag.ingest")


def ingest_pdf(file_path: str, original_filename: str) -> Tuple[int, int]:
    """
    Ingest a PDF file into the vector store.
    Splits into chunks, adds metadata (source, page), and stores vectors in Chroma.

    Returns: (chunks_created, total_pages)
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    total_pages = len(pages)
    if total_pages == 0:
        logger.warning(f"File {original_filename} yielded 0 pages.")
        return 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    documents = []
    for doc in pages:
        raw_page = doc.metadata.get("page", 0)
        doc.metadata["source"] = original_filename
        doc.metadata["page"] = int(raw_page) + 1  # 1-based page index
        doc.metadata["page_1_indexed"] = True
        documents.append(doc)

    chunks = splitter.split_documents(documents)

    if chunks:
        vectorstore = get_vectorstore()
        vectorstore.add_documents(chunks)
        logger.info(
            f"Ingested {len(chunks)} chunks from '{original_filename}' ({total_pages} pages)."
        )

    return len(chunks), total_pages
