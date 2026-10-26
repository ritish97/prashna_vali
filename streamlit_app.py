"""
Streamlit Interactive Live Demo for RAG Pipeline.
Powered by Groq (openai/gpt-oss-20b), ChromaDB, and Local HuggingFace Embeddings.

Run locally:
    streamlit run streamlit_app.py

Deploy on Render:
    streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
"""

import logging
import os
import shutil
import statistics
import sys
import time
import warnings
from pathlib import Path

# Suppress harmless module inspection warnings from Streamlit watcher
warnings.filterwarnings("ignore")
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.CRITICAL)

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="RAG Pipeline Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .citation-box {
        background-color: #020617;
        border-left: 3px solid #38bdf8;
        padding: 10px 14px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    .badge-hit {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-miss {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Lazy-import backend components
try:
    from app.config import settings
    from app.ingest import ingest_pdf
    from app.llm import generate_answer
    from app.retrieval import _query_cache, retrieve
    from app.vectorstore import (
        delete_document_by_name,
        get_vectorstore,
        list_ingested_documents,
    )
except Exception as e:
    st.error(f"Error loading backend components: {e}")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_metrics" not in st.session_state:
    st.session_state.last_query_metrics = None


# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Settings")

    # API Key Configuration
    current_key = settings.get_active_api_key() or ""
    groq_api_key = st.text_input(
        "Groq API Key",
        value=current_key,
        type="password",
        help="Enter your Groq API key (starts with gsk_). Can also be configured via GROQ_API_KEY environment variable.",
    )
    if groq_api_key:
        settings.GROQ_API_KEY = groq_api_key
        os.environ["GROQ_API_KEY"] = groq_api_key

    # Model and Retrieval params
    model_name = st.selectbox(
        "LLM Model",
        options=["openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
        help="Primary model served on Groq.",
    )
    settings.LLM_MODEL = model_name

    top_k = st.slider("Context Chunks (Top-K)", min_value=1, max_value=8, value=settings.TOP_K)
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=settings.LLM_TEMPERATURE, step=0.1)
    settings.LLM_TEMPERATURE = temperature

    st.markdown("---")
    st.markdown("### 📂 Document Ingestion")

    # Fast Sample Contract Button (auto-generate if missing)
    sample_pdf_path = ROOT_DIR / "sample_contract.pdf"
    if not sample_pdf_path.exists():
        try:
            from scripts.generate_sample_pdf import generate_sample_pdf
            generate_sample_pdf(str(sample_pdf_path))
        except Exception:
            pass

    if sample_pdf_path.exists():
        if st.button("📄 Load Sample Contract", use_container_width=True):
            with st.spinner("Indexing sample_contract.pdf..."):
                try:
                    chunks, pages = ingest_pdf(str(sample_pdf_path), "sample_contract.pdf")
                    st.success(f"✓ Indexed {chunks} chunks across {pages} page(s)!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Failed to index: {ex}")

    # File uploader
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    if uploaded_file is not None:
        if st.button("⬆️ Ingest Uploaded PDF", use_container_width=True):
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                upload_dir = Path(settings.UPLOAD_DIR)
                upload_dir.mkdir(parents=True, exist_ok=True)
                save_dest = upload_dir / uploaded_file.name

                with open(save_dest, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    chunks, pages = ingest_pdf(str(save_dest), uploaded_file.name)
                    st.success(f"✓ Ingested {chunks} chunks across {pages} page(s)!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Ingestion failed: {ex}")

    st.markdown("---")
    st.markdown("### 📚 Indexed Documents")
    try:
        docs = list_ingested_documents()
        if not docs:
            st.caption("No documents indexed yet. Upload a PDF or click 'Load Sample Contract' above.")
        else:
            for d in docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{d['filename']}** ({d['chunk_count']} chunks)")
                with col2:
                    if st.button("🗑️", key=f"del_{d['filename']}", help=f"Delete {d['filename']}"):
                        delete_document_by_name(d["filename"])
                        st.rerun()

            if st.button("Clear All Documents", use_container_width=True):
                for d in docs:
                    delete_document_by_name(d["filename"])
                st.session_state.messages = []
                st.rerun()
    except Exception as e:
        st.caption(f"Status: {e}")


# --- MAIN INTERFACE TABS ---
tab_chat, tab_benchmark, tab_arch = st.tabs(["💬 Interactive Q&A", "⚡ Latency Benchmark", "🏛️ Architecture & Notes"])


# ==========================================
# TAB 1: INTERACTIVE CHAT
# ==========================================
with tab_chat:
    st.markdown('<div class="main-header">RAG Pipeline Studio</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Grounded Document Q&A &middot; Powered by <strong>Groq</strong> ({settings.LLM_MODEL}) &middot; Local Embeddings &middot; Query-Cache</div>',
        unsafe_allow_html=True,
    )

    # Suggestion Pills
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_q = None
    with col_p1:
        if st.button("📌 What is the termination clause?"):
            preset_q = "What is the termination clause in this contract?"
    with col_p2:
        if st.button("💳 What are the payment terms and late fees?"):
            preset_q = "What are the payment terms and late fees?"
    with col_p3:
        if st.button("🔒 What are the data protection requirements?"):
            preset_q = "What are the data confidentiality and encryption requirements?"

    # Initial guidance if no messages yet
    if not st.session_state.messages:
        if not docs:
            st.info("👋 **Welcome to Prashnavali!** No documents are indexed yet.\n\nClick **'📄 Load Sample Contract'** in the sidebar (or upload any PDF) to get started immediately.")
        else:
            st.info(f"📚 **{len(docs)} document(s) indexed and ready.** Use the suggestion buttons above or type a question in the chat bar below.")

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metrics"):
                m = msg["metrics"]
                badge_class = "badge-hit" if m["cache_hit"] else "badge-miss"
                badge_text = "⚡ CACHE HIT" if m["cache_hit"] else "🐢 CACHE MISS"
                st.markdown(
                    f"""
                    <div style="display: flex; gap: 12px; align-items: center; font-size: 0.8rem; color: #94a3b8; margin-top: 8px;">
                        <span class="{badge_class}">{badge_text}</span>
                        <span>Retrieval: <strong>{m['retrieval_ms']:.2f} ms</strong></span>
                        <span>&bull;</span>
                        <span>LLM Generation: <strong>{m['generation_ms']:.2f} ms</strong></span>
                        <span>&bull;</span>
                        <span>Total: <strong>{(m['retrieval_ms'] + m['generation_ms']):.2f} ms</strong></span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if msg.get("sources"):
                with st.expander(f"📑 Grounding Sources ({len(msg['sources'])} chunks)", expanded=False):
                    for idx, src in enumerate(msg["sources"], 1):
                        score_txt = f" &middot; Relevance: {src['score']}" if src.get("score") is not None else ""
                        st.markdown(f"**Chunk {idx}: {src['file_name']} (Page {src['page']}){score_txt}**")
                        st.markdown(f"<div class='citation-box'>{src['content']}</div>", unsafe_allow_html=True)

    # Handle Input (Chat bar or preset button)
    user_query = st.chat_input("Ask a question about your documents...") or preset_q

    if user_query:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate grounded answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving semantic chunks & querying Groq..."):
                try:
                    chunks, retrieval_ms, cache_hit = retrieve(user_query, top_k=top_k)
                    answer, generation_ms = generate_answer(user_query, chunks)

                    st.markdown(answer)

                    # Metrics Display
                    badge_class = "badge-hit" if cache_hit else "badge-miss"
                    badge_text = "⚡ CACHE HIT" if cache_hit else "🐢 CACHE MISS"
                    st.markdown(
                        f"""
                        <div style="display: flex; gap: 12px; align-items: center; font-size: 0.8rem; color: #94a3b8; margin-top: 8px;">
                            <span class="{badge_class}">{badge_text}</span>
                            <span>Retrieval: <strong>{retrieval_ms:.2f} ms</strong></span>
                            <span>&bull;</span>
                            <span>LLM Generation: <strong>{generation_ms:.2f} ms</strong></span>
                            <span>&bull;</span>
                            <span>Total: <strong>{(retrieval_ms + generation_ms):.2f} ms</strong></span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Source Citations
                    if chunks:
                        with st.expander(f"📑 Grounding Sources ({len(chunks)} chunks)", expanded=True):
                            for idx, src in enumerate(chunks, 1):
                                score_txt = f" &middot; Relevance: {src.score}" if src.score is not None else ""
                                st.markdown(f"**Chunk {idx}: {src.file_name} (Page {src.page}){score_txt}**")
                                st.markdown(f"<div class='citation-box'>{src.content}</div>", unsafe_allow_html=True)

                    # Save to session history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metrics": {
                            "retrieval_ms": retrieval_ms,
                            "generation_ms": generation_ms,
                            "cache_hit": cache_hit,
                        },
                        "sources": [
                            {
                                "file_name": c.file_name,
                                "page": c.page,
                                "content": c.content,
                                "score": c.score,
                            }
                            for c in chunks
                        ],
                    })

                except Exception as ex:
                    st.error(f"Query processing error: {ex}")


# ==========================================
# TAB 2: LATENCY BENCHMARK
# ==========================================
with tab_benchmark:
    st.markdown("### ⚡ Live Query-Embedding Cache Benchmark")
    st.write(
        "Demonstrates the empirical latency improvement of caching query embeddings in an LRU cache. "
        "The first call computes the vector via CPU sentence-transformers (Cold Start), "
        "while subsequent calls retrieve the pre-computed embedding in memory (Cache Hit)."
    )

    bench_q = st.text_input(
        "Benchmark Query",
        value="What is the termination clause in this contract?",
        key="bench_q",
    )
    iterations = st.slider("Warm Cache Iterations", min_value=5, max_value=30, value=15)

    if st.button("🚀 Run Cache Benchmark", type="primary"):
        with st.spinner("Benchmarking retrieval latency..."):
            try:
                # Clear cache for honest cold measurement
                _query_cache._store.clear()

                # Cold call
                _, cold_ms, cache_hit_cold = retrieve(bench_q)

                # Warm calls
                warm_times = []
                for _ in range(iterations):
                    _, w_ms, cache_hit_warm = retrieve(bench_q)
                    warm_times.append(w_ms)

                warm_avg = statistics.mean(warm_times)
                improvement = ((cold_ms - warm_avg) / cold_ms) * 100 if cold_ms > 0 else 0.0

                st.success("Benchmark completed successfully!")

                # Metric Cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cold Retrieval (Miss)", f"{cold_ms:.2f} ms")
                with col2:
                    st.metric(f"Warm Avg ({iterations} runs)", f"{warm_avg:.2f} ms")
                with col3:
                    st.metric("Latency Reduction", f"{improvement:.1f}%", delta=f"{improvement:.1f}%")

                # Bar comparison
                st.bar_chart({"Cold Start (Miss) [ms]": [cold_ms], "Warm Cache (Hit) [ms]": [warm_avg]})

            except Exception as e:
                st.error(f"Benchmark error: {e}. Please ensure at least one document is indexed.")


# ==========================================
# TAB 3: ARCHITECTURE & RESUME BULLETS
# ==========================================
with tab_arch:
    st.markdown("### 🏛️ Architecture & Talking Points")
    st.markdown(
        """
        #### 1. Hallucination Mitigation
        - **Strict Grounding:** The system prompt prohibits the LLM from extrapolating or using outside knowledge.
        - **Citations:** Every factual statement cites its exact source file and 1-based page number.
        - **Fallback:** If context is insufficient, the model explicitly responds: *"I cannot find the answer to this question in the provided documents."*

        #### 2. Local Vector Processing
        - Embeddings are generated locally using `sentence-transformers/all-MiniLM-L6-v2`.
        - Documents remain completely local during parsing and indexing—only the final retrieved context chunks are sent to the LLM.

        #### 3. In-Memory Query Caching
        - Production traffic often includes repeated or FAQ-style questions.
        - By caching the **query embedding** (rather than the static answer), answers remain dynamic while skipping the most CPU-intensive step.
        """
    )
