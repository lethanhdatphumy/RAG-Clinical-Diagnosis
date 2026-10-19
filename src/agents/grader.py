"""
Specificity Grader Node.

Decides whether a query has enough clinical detail to go straight to
Hybrid RAG, or needs the Enrichment Agent first.

A query is "specific enough" when it contains at least 2 of:
  - 3+ distinct symptoms
  - demographic info (age / gender)
  - geographic / travel context
  - lab findings or vital signs
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from config.settings import Config
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

_GRADER_TEMPLATE = """You are a clinical triage specialist.

Decide whether the query below contains ENOUGH clinical detail to search a case database directly.

A query is SPECIFIC ENOUGH if it includes at least 2 of:
- 3 or more distinct symptoms
- Patient demographics (age or gender)
- Geographic location or travel history
- Lab results or vital signs

Query: {query}

Respond with EXACTLY one line:
SPECIFIC: <yes|no> | REASON: <one sentence>"""

_GRADER_PROMPT = PromptTemplate(template=_GRADER_TEMPLATE, input_variables=["query"])


def specificity_grader_node(state: AgentState) -> AgentState:
    """
    Sets state["is_specific"] = True/False.
    Falls back to False (enricher path) on any error.
    """
    query = state["query"]
    path = list(state.get("path_taken") or []) + ["specificity_grader"]
    logger.info("Specificity grader — query: %s", query[:80])

    try:
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.0,
            max_output_tokens=512,
        )
        response = llm.invoke(_GRADER_PROMPT.format(query=query))
        content = response.content
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = text_parts[0] if text_parts else ""
        raw = content.strip()
        logger.info("Grader raw response: %s", raw)
        is_specific = _parse(raw)
    except Exception as exc:
        logger.error("Specificity grader error: %s — defaulting to enricher path", exc)
        is_specific = False

    logger.info("Query is specific: %s", is_specific)
    return {**state, "is_specific": is_specific, "path_taken": path}


def _parse(raw: str) -> bool:
    try:
        verdict = raw.split("|")[0].split(":")[1].strip().lower()
        return verdict == "yes"
    except Exception:
        return False
