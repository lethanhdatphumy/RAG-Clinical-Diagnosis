import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from config.settings import Config
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

_ROUTER_TEMPLATE = """You are a medical AI routing agent.

Your job is to decide whether a user query should be answered by:
- "rag": Retrieval-Augmented Generation — best for specific clinical case queries that
  describe a real patient (symptoms, travel history, lab results, vital signs, etc.).
  RAG retrieves similar documented cases from a medical database to ground the answer.
- "finetuned": A fine-tuned clinical LLM — best for general medical knowledge questions
  about a disease (definition, epidemiology, standard treatment protocols, pathophysiology)
  where case-level retrieval adds little value.

Rules:
1. If the query describes an individual patient with symptoms or clinical findings → "rag"
2. If the query asks general knowledge about a disease or tropical medicine → "finetuned"
3. When in doubt, prefer "rag" for patient-safety reasons.

Query: {query}

Respond with EXACTLY one line in this format (no other text):
ROUTE: <rag|finetuned> | REASON: <one sentence>
"""

_ROUTER_PROMPT = PromptTemplate(
    template=_ROUTER_TEMPLATE,
    input_variables=["query"],
)


def router_node(state: AgentState) -> AgentState:
    """
    Classifies the query and sets state["route"] to "rag" or "finetuned".
    Falls back to "rag" on any parsing or API error.
    """
    query = state["query"]
    logger.info("Router node — classifying query: %s", query[:80])

    try:
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GOOGLE_API_KEY,
            temperature=0.0,
            max_output_tokens=64,
        )
        prompt_text = _ROUTER_PROMPT.format(query=query)
        response = llm.invoke(prompt_text)
        raw = response.content.strip()
        logger.info("Router raw response: %s", raw)

        route, reasoning = _parse_router_response(raw)
    except Exception as exc:
        logger.error("Router node error: %s — defaulting to rag", exc)
        route = "rag"
        reasoning = f"Routing failed ({exc}); defaulting to RAG for safety."

    logger.info("Routing decision: %s", route)
    return {
        **state,
        "route": route,
        "route_reasoning": reasoning,
    }


def _parse_router_response(raw: str) -> tuple[str, str]:
    """
    Parse 'ROUTE: rag | REASON: ...' into (route, reasoning).
    Falls back to ("rag", raw) if the format is unexpected.
    """
    try:
        parts = raw.split("|", 1)
        route_part = parts[0].split(":", 1)[1].strip().lower()
        reason_part = parts[1].split(":", 1)[1].strip() if len(parts) > 1 else raw

        if route_part not in ("rag", "finetuned"):
            raise ValueError(f"Unknown route value: {route_part!r}")

        return route_part, reason_part
    except Exception:
        return "rag", raw
