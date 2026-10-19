"""
Enricher Node (Agentic).

For vague queries, the enricher autonomously:
  1. Infers plausible clinical context (age, region, risk factors) from the symptom pattern.
  2. Performs a web search to ground those assumptions.
  3. Produces an enriched_query ready for Hybrid RAG.

The enricher never asks the user follow-up questions — it reasons and acts on its own.
"""

import logging
import re

from google import genai

from config.settings import Config
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

# ── Prompts ───────────────────────────────────────────────────────────────────

_INFER_PROMPT = """You are an autonomous clinical reasoning agent.

A patient described their symptoms vaguely. Infer the most likely clinical context.

Vague query: {query}

Reason briefly, then output EXACTLY this block (no extra text):
ENRICHED: <rewritten query with inferred context, 1–2 sentences>
ASSUMPTIONS:
- <assumption 1>
- <assumption 2>
- <assumption 3>"""

_SEARCH_SUMMARY_PROMPT = """You are a medical information assistant.

Given the web search results below, extract facts relevant to this clinical query.
Summarise in 2–3 bullet points. Be factual; do not diagnose.

Query: {query}
Search results:
{results}

Summary:"""

# ── Web search (DuckDuckGo, no API key needed) ────────────────────────────────

def _web_search(query: str, max_results: int = 3) -> str:
    """Run a simple DuckDuckGo instant-answer search. Returns a text summary."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        if not hits:
            return ""
        lines = [f"- {h.get('title','')}: {h.get('body','')}" for h in hits]
        return "\n".join(lines)
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return ""


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_gemini(prompt: str) -> str:
    client = genai.Client(api_key=Config.GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=Config.GEMINI_MODEL,
        contents=prompt,
    )
    return response.text.strip() if response.text else ""


# ── Parsers ───────────────────────────────────────────────────────────────────

def _parse_enriched(raw: str) -> tuple[str, list[str]]:
    """Extract ENRICHED: … and ASSUMPTIONS: … from the LLM response."""
    enriched_query = ""
    assumptions = []

    enriched_match = re.search(r"ENRICHED:\s*(.+?)(?:\n|$)", raw, re.IGNORECASE)
    if enriched_match:
        enriched_query = enriched_match.group(1).strip().strip('"')

    assumptions_block = re.search(r"ASSUMPTIONS:(.*)", raw, re.IGNORECASE | re.DOTALL)
    if assumptions_block:
        for line in assumptions_block.group(1).splitlines():
            line = line.strip().lstrip("-•* ").strip()
            if line:
                assumptions.append(line)

    return enriched_query, assumptions


# ── Node ──────────────────────────────────────────────────────────────────────

def enricher_node(state: AgentState) -> AgentState:
    """
    Autonomously enriches a vague query.
    Sets enriched_query, assumptions, and web_search_results in state.
    Falls back to the original query on any error.
    """
    query = state["query"]
    path = list(state.get("path_taken") or []) + ["enricher"]
    logger.info("Enricher — original query: %s", query[:80])

    enriched_query = query
    assumptions: list[str] = []
    web_search_results = ""

    try:
        # Step 1 — infer clinical context
        raw = _call_gemini(_INFER_PROMPT.format(query=query))
        logger.debug("Enricher raw response:\n%s", raw)
        enriched_query, assumptions = _parse_enriched(raw)

        if not enriched_query:
            logger.warning("Enricher could not parse ENRICHED — using original query")
            enriched_query = query

        # Step 2 — web search to ground the assumptions
        search_query = enriched_query or query
        raw_results = _web_search(search_query)
        if raw_results:
            summary = _call_gemini(
                _SEARCH_SUMMARY_PROMPT.format(query=search_query, results=raw_results)
            )
            web_search_results = summary or raw_results

        logger.info("Enricher — enriched query: %s", enriched_query[:80])
        logger.info("Enricher — %d assumptions, web results: %s chars",
                    len(assumptions), len(web_search_results))

    except Exception as exc:
        logger.error("Enricher error: %s — using original query", exc)
        enriched_query = query

    return {
        **state,
        "enriched_query": enriched_query,
        "assumptions": assumptions,
        "web_search_results": web_search_results,
        "path_taken": path,
    }
