import logging
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.router import router_node
from src.agents.rag_node import rag_node
from src.agents.finetuned_node import finetuned_node

logger = logging.getLogger(__name__)


def _route_decision(state: AgentState) -> str:
    """Conditional edge: directs flow based on the router's decision."""
    route = state.get("route", "rag")
    logger.info("Graph edge — routing to: %s", route)
    return route


def build_clinical_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph for clinical diagnosis routing.

    Graph topology:
        router → (rag | finetuned) → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("rag", rag_node)
    graph.add_node("finetuned", finetuned_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_decision,
        {
            "rag": "rag",
            "finetuned": "finetuned",
        },
    )

    graph.add_edge("rag", END)
    graph.add_edge("finetuned", END)

    return graph.compile()


# Module-level compiled graph (lazy-initialised on first import)
clinical_graph = build_clinical_graph()


def run_diagnosis(query: str) -> AgentState:
    """
    Public entry point: run the full agentic pipeline for a user query.

    Returns the final AgentState containing route, answer, model_used, etc.
    """
    initial_state: AgentState = {
        "query": query,
        "route": None,
        "route_reasoning": None,
        "answer": None,
        "source_documents": None,
        "model_used": None,
        "error": None,
    }

    logger.info("Running clinical graph for query: %s", query[:80])
    final_state = clinical_graph.invoke(initial_state)
    return final_state
