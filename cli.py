"""
Interactive Command-Line Interface (CLI) for the RAG Pipeline.
Allows terminal-based ingestion, querying, and interactive chat.

Usage:
    python cli.py ingest sample_contract.pdf
    python cli.py query "What are the payment terms?"
    python cli.py interactive (or python cli.py)
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import settings
from app.ingest import ingest_pdf
from app.llm import generate_answer
from app.retrieval import retrieve
from app.vectorstore import get_vectorstore, list_ingested_documents


def print_banner():
    print("=" * 65)
    print("       RAG Pipeline Terminal Interface")
    print(f"       Model: {settings.LLM_MODEL} | Provider: Groq")
    print("=" * 65)


def do_ingest(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"[Error] File not found: {file_path}")
        return False
    if not path.suffix.lower() == ".pdf":
        print(f"[Error] Only PDF files are supported: {file_path}")
        return False

    print(f"[*] Ingesting '{path.name}' into ChromaDB...")
    try:
        chunks, pages = ingest_pdf(str(path.resolve()), path.name)
        print(f"[+] Success: Created {chunks} chunks across {pages} page(s).")
        return True
    except Exception as e:
        print(f"[!] Ingestion failed: {e}")
        return False


def do_query(question: str):
    print(f"\n[?] Question: {question}")
    print("[*] Retrieving context chunks & querying Groq (openai/gpt-oss-20b)...")

    try:
        chunks, retrieval_ms, cache_hit = retrieve(question)
        answer, generation_ms = generate_answer(question, chunks)

        cache_label = "[CACHE HIT]" if cache_hit else "[CACHE MISS]"
        total_ms = retrieval_ms + generation_ms

        print("\n" + "-" * 65)
        print("  ANSWER:")
        print("-" * 65)
        print(answer.strip())
        print("-" * 65)

        print("\n[Metrics]")
        print(f"  Status        : {cache_label}")
        print(f"  Retrieval     : {retrieval_ms:.2f} ms")
        print(f"  LLM Generation: {generation_ms:.2f} ms")
        print(f"  Total Latency : {total_ms:.2f} ms")

        if chunks:
            print("\n[Source Citations]")
            for i, c in enumerate(chunks, 1):
                score_str = f" (relevance: {c.score})" if c.score is not None else ""
                snippet = c.content.replace("\n", " ").strip()
                if len(snippet) > 120:
                    snippet = snippet[:120] + "..."
                print(f"  {i}. [{c.file_name} | Page {c.page}]{score_str}")
                print(f"     \"{snippet}\"")
        print("-" * 65 + "\n")

    except Exception as e:
        print(f"[!] Query failed: {e}")


def do_list_documents():
    docs = list_ingested_documents()
    print("\n--- Ingested Documents ---")
    if not docs:
        print("  (No documents ingested yet. Use 'ingest <path.pdf>')")
    else:
        for d in docs:
            print(f"  * {d['filename']} ({d['chunk_count']} chunks)")
    print("--------------------------\n")


def interactive_loop():
    print_banner()
    print("Commands:")
    print("  <question>           - Ask any question against your ingested documents")
    print("  ingest <path.pdf>    - Ingest a PDF file into the vector store")
    print("  docs                 - List currently ingested documents")
    print("  help                 - Show this help message")
    print("  exit / quit          - Exit terminal interface")
    print("-" * 65 + "\n")

    while True:
        try:
            user_input = input("rag> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue

        cmd_lower = user_input.lower()
        if cmd_lower in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        elif cmd_lower in ("help", "?"):
            print("Commands: <question>, ingest <path.pdf>, docs, exit")
        elif cmd_lower == "docs":
            do_list_documents()
        elif cmd_lower.startswith("ingest "):
            file_arg = user_input[7:].strip().strip("'\"")
            do_ingest(file_arg)
        else:
            do_query(user_input)


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline Terminal Interface")
    subparsers = parser.add_subparsers(dest="command")

    # Ingest subcommand
    ingest_p = subparsers.add_parser("ingest", help="Ingest a PDF file")
    ingest_p.add_argument("pdf_path", type=str, help="Path to PDF file")

    # Query subcommand
    query_p = subparsers.add_parser("query", help="Ask a question")
    query_p.add_argument("question", type=str, help="Question to ask")

    # Docs subcommand
    subparsers.add_parser("docs", help="List ingested documents")

    # Interactive subcommand
    subparsers.add_parser("interactive", help="Start interactive chat shell")

    args = parser.parse_args()

    if args.command == "ingest":
        do_ingest(args.pdf_path)
    elif args.command == "query":
        do_query(args.question)
    elif args.command == "docs":
        do_list_documents()
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
