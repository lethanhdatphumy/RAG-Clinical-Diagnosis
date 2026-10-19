"""
LangGraph — Adaptive + Hybrid + Agentic RAG pipeline.

Full topology:

    router ─┬─► finetuned ─────────────────────────────────────────► END
            │
            └─► specificity_grader ─┬─► hybrid_retriever ─► generator ─► END
                  (is_specific?)    │
                                    └─► enricher ──► hybrid_retriever ─► generator ─► END

Adaptive  : router decides rag vs fine-tuned based on query type.
Hybrid    : hybrid_retriever fuses BM25 + Qdrant dense via RRF.
Agentic   : enricher autonomously infers context & runs web search for vague queries.
"""

import logging
from langgraph.graph import StateGraph, END

from src.agents.state import AgentState
from src.agents.router import router_node
from src.agents.finetuned_node import finetuned_node
from src.agents.grader import specificity_grader_node
from src.agents.enricher import enricher_node
from src.agents.hybrid_retriever import hybrid_retriever_node
from src.agents.answer_grader import answer_grader_node
from src.agents.generator import generator_node

logger = logging.getLogger(__name__)


# ── Conditional edge functions ────────────────────────────────────────────────

def _route_decision(state: AgentState) -> str:
    route = state.get("route", "rag")
    logger.info("Graph — adaptive route: %s", route)
    return route


def _specificity_decision(state: AgentState) -> str:
    is_specific = state.get("is_specific", False)
    if is_specific:
        decision = "specific"
    elif state.get("enriched_query"):
        decision = "user_context"   # user already provided context — skip enricher
    else:
        decision = "vague"
    logger.info("Graph — specificity decision: %s", decision)
    return decision


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_clinical_graph() -> StateGraph:
    """
    Compiles the full Adaptive + Hybrid + Agentic RAG pipeline.
    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("router", router_node)
    graph.add_node("finetuned", finetuned_node)
    graph.add_node("specificity_grader", specificity_grader_node)
    graph.add_node("enricher", enricher_node)
    graph.add_node("hybrid_retriever", hybrid_retriever_node)
    graph.add_node("answer_grader", answer_grader_node)
    graph.add_node("generator", generator_node)

    # Entry point
    graph.set_entry_point("router")

    # Adaptive routing: router → finetuned | specificity_grader
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {
            "rag": "specificity_grader",
            "finetuned": "finetuned",
        },
    )

    # Agentic routing: specificity_grader → hybrid_retriever | enricher
    graph.add_conditional_edges(
        "specificity_grader",
        _specificity_decision,
        {
            "specific": "hybrid_retriever",       # enough detail → retrieve directly
            "user_context": "hybrid_retriever",   # user added context → skip enricher
            "vague": "enricher",                  # too vague → autonomous enrichment
        },
    )

    # Enricher always flows into retrieval
    graph.add_edge("enricher", "hybrid_retriever")

    # Retrieval → answer grader → generator or fine-tuned fallback
    graph.add_edge("hybrid_retriever", "answer_grader")
    graph.add_conditional_edges(
        "answer_grader",
        lambda s: "generator" if s.get("docs_relevant") else "finetuned",
        {
            "generator": "generator",
            "finetuned": "finetuned",
        },
    )
    graph.add_edge("generator", END)

    # Fine-tuned path ends directly (direct route + RAG fallback)
    graph.add_edge("finetuned", END)

    return graph.compile()


# ── Module-level compiled graph ───────────────────────────────────────────────

clinical_graph = build_clinical_graph()


# ── Public entry point ────────────────────────────────────────────────────────

def run_diagnosis(query: str, user_context: str | None = None) -> AgentState:
    """
    Run the full Adaptive + Hybrid + Agentic RAG pipeline for a user query.

    Args:
        query:        The user's original query.
        user_context: Optional extra context the user provided when prompted
                      (e.g. age, travel history, duration of symptoms).
                      If supplied, it is concatenated with the query and the
                      autonomous enricher is skipped.
                      Pass None to let the enricher run autonomously.

    Returns:
        The final AgentState (answer, source_documents, path_taken, …).
    """
    enriched_query = None
    if user_context and user_context.strip():
        enriched_query = f"{query.strip()} {user_context.strip()}"
        logger.info("User context provided — enriched query: %s", enriched_query[:120])

    initial_state: AgentState = {
        "query": query,
        "route": None,
        "route_reasoning": None,
        "is_specific": None,
        "enriched_query": enriched_query,
        "assumptions": None,
        "web_search_results": None,
        "source_documents": None,
        "answer": None,
        "model_used": None,
        "path_taken": [],
        "error": None,
    }

    logger.info("Running clinical graph — query: %s", query[:80])
    return clinical_graph.invoke(initial_state)
