"""
Generator Node.

Takes retrieved source_documents and produces a final clinical answer.
Uses Gemini to synthesise the answer from the case evidence.
"""

import logging

from google import genai

from config.settings import Config
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = """You are a clinical diagnosis assistant.

Based on the retrieved case reports below, answer the patient's question.

Rules:
- Cite the most relevant case(s) by their case_id.
- Always recommend consulting a qualified clinician for any personal medical decision.
- If the retrieved cases are not relevant, say so honestly.

Retrieved cases:
{context}

Patient question: {query}

Answer:"""


def _format_docs(docs: list[dict]) -> str:
    if not docs:
        return "No relevant cases retrieved."
    parts = []
    for doc in docs:
        case_id = doc.get("case_id", "unknown")
        content = doc.get("page_content", "")
        diseases = ", ".join(doc.get("diseases", []))
        parts.append(f"[{case_id}] Diseases: {diseases}\n{content[:800]}")
    return "\n\n---\n\n".join(parts)


def generator_node(state: AgentState) -> AgentState:
    """
    Generates the final answer from retrieved documents.
    Falls back to a no-retrieval message if docs are empty.
    """
    query = state.get("enriched_query") or state["query"]
    docs = state.get("source_documents") or []
    path = list(state.get("path_taken") or []) + ["generator"]
    logger.info("Generator — %d source docs, query: %s", len(docs), query[:80])

    try:
        context = _format_docs(docs)
        prompt = _GENERATOR_PROMPT.format(context=context, query=query)

        client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=prompt,
        )
        answer = response.text.strip() if response.text else ""

        return {
            **state,
            "answer": answer,
            "model_used": f"RAG + {Config.GEMINI_MODEL}",
            "path_taken": path,
        }

    except Exception as exc:
        logger.error("Generator error: %s", exc)
        return {
            **state,
            "answer": None,
            "error": str(exc),
            "model_used": f"RAG + {Config.GEMINI_MODEL}",
            "path_taken": path,
        }
