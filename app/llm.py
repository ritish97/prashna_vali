import logging
import time
from typing import List, Tuple

from app.config import settings
from app.models import SourceChunk

logger = logging.getLogger("rag.llm")

SYSTEM_PROMPT = """You are a precise, grounded document assistant.
Your answers MUST be strictly derived from the provided document excerpts.

Guidelines:
1. Grounding: Answer strictly using facts stated in the provided context. Do NOT use outside knowledge or extrapolate.
2. Missing Info: If the context does not contain sufficient information to answer the question, state clearly: "I cannot find the answer to this question in the provided documents."
3. Inline Citations: Whenever you state a fact, cite the source document and page number in brackets, for example: [Document: filename.pdf, Page: X].
4. Clarity: Provide concise, professional, and well-structured answers.
"""


def generate_answer(
    question: str, context_chunks: List[SourceChunk]
) -> Tuple[str, float]:
    """
    Generate an answer using the LLM grounded strictly in the provided chunks.
    Supports Groq (with openai/gpt-oss-20b) as well as OpenAI / Ollama.
    Returns: (answer_text, generation_latency_ms)
    """
    t0 = time.perf_counter()

    if not context_chunks:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return (
            "No relevant document chunks were found to answer your question. "
            "Please make sure documents have been ingested.",
            elapsed_ms,
        )

    # Format context blocks
    context_str = "\n\n---\n\n".join(
        [
            f"[Source: {chunk.file_name}, Page: {chunk.page}]\n{chunk.content}"
            for chunk in context_chunks
        ]
    )

    user_prompt = f"""Context:
{context_str}

Question: {question}

Answer based ONLY on the context above:"""

    api_key = settings.get_active_api_key()
    base_url = settings.OPENAI_BASE_URL

    if not api_key and not base_url:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        sources_summary = ", ".join(
            [f"{c.file_name} (p. {c.page})" for c in context_chunks]
        )
        preview = "\n\n".join(
            [
                f"> **{c.file_name} (Page {c.page})**:\n> {c.content[:200]}..."
                for c in context_chunks
            ]
        )
        return (
            f"[Note: LLM API key not configured. Set GROQ_API_KEY in .env]\n\n"
            f"Successfully retrieved {len(context_chunks)} relevant chunk(s) from {sources_summary}:\n\n"
            f"{preview}",
            elapsed_ms,
        )

    try:
        if settings.is_groq():
            # Native Groq client execution
            from groq import Groq

            client = Groq(api_key=api_key)

            # Build request kwargs
            kwargs = {
                "model": settings.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": settings.LLM_TEMPERATURE,
                "max_completion_tokens": settings.LLM_MAX_TOKENS,
                "top_p": settings.LLM_TOP_P,
                "stream": True,
                "stop": None,
            }

            # reasoning_effort applies to reasoning models like openai/gpt-oss-20b
            if settings.REASONING_EFFORT and "gpt-oss" in settings.LLM_MODEL.lower():
                kwargs["reasoning_effort"] = settings.REASONING_EFFORT

            completion = client.chat.completions.create(**kwargs)

            chunks = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content or ""
                    chunks.append(content)
            answer = "".join(chunks)

        else:
            # Standard OpenAI / LangChain provider
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                api_key=api_key or "placeholder",
                base_url=base_url,
            )
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            answer = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )

    except Exception as e:
        logger.error(f"Error calling LLM ({settings.LLM_MODEL}): {e}")
        answer = f"Error generating answer from LLM: {str(e)}"

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return answer, elapsed_ms
