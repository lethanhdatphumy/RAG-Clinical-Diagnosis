"""
Answer Grader Node.

Checks whether the documents retrieved from Qdrant are actually relevant
to the user's query. If none are relevant, the pipeline falls back to the
fine-tuned LLM instead of generating a hallucinated answer from poor docs.
"""

import logging

from google import genai

from config.settings import Config
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

_GRADER_PROMPT = """You are a clinical information retrieval specialist.

Below is a patient query and a set of case reports retrieved from a clinical database.
Decide whether ANY of the retrieved cases contain information genuinely relevant to the query.

A case is relevant if it shares at least one of:
- The same primary disease or disease category
- Overlapping key symptoms
- Similar demographic or geographic context

Query: {query}

Retrieved cases:
{cases}

Respond with EXACTLY one line:
RELEVANT: <yes|no> | REASON: <one sentence>"""


def _format_cases(docs: list[dict]) -> str:
    if not docs:
        return "No cases retrieved."
    parts = []
    for doc in docs:
        case_id = doc.get("case_id", "unknown")
        diseases = ", ".join(doc.get("diseases", [])) or "unknown"
        symptoms = ", ".join(doc.get("symptoms", [])) or "unknown"
        parts.append(f"[{case_id}] Diseases: {diseases} | Symptoms: {symptoms}")
    return "\n".join(parts)


def answer_grader_node(state: AgentState) -> AgentState:
    """
    Sets state["docs_relevant"] = True/False.
    Defaults to False (fine-tuned fallback) on any error or empty docs.
    """
    query = state.get("enriched_query") or state["query"]
    docs = state.get("source_documents") or []
    path = list(state.get("path_taken") or []) + ["answer_grader"]

    if not docs:
        logger.info("Answer grader — no docs retrieved, routing to fine-tuned LLM")
        return {**state, "docs_relevant": False, "path_taken": path}

    logger.info("Answer grader — grading %d docs for query: %s", len(docs), query[:80])

    try:
        cases_text = _format_cases(docs)
        prompt = _GRADER_PROMPT.format(query=query, cases=cases_text)

        client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=Config.GEMINI_MODEL,
            contents=prompt,
        )
        raw = response.text.strip() if response.text else ""
        logger.info("Answer grader raw response: %s", raw)
        docs_relevant = _parse(raw)

    except Exception as exc:
        logger.error("Answer grader error: %s — defaulting to fine-tuned fallback", exc)
        docs_relevant = False

    logger.info("Docs relevant: %s", docs_relevant)
    return {**state, "docs_relevant": docs_relevant, "path_taken": path}


def _parse(raw: str) -> bool:
    try:
        verdict = raw.split("|")[0].split(":")[1].strip().lower()
        return verdict == "yes"
    except Exception:
        return False
