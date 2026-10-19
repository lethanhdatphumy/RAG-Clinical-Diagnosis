from typing import TypedDict, Optional


class AgentState(TypedDict):
    """Shared state that flows through every node of the LangGraph."""

    # ── Input ─────────────────────────────────────────────────────────────────
    query: str                        # Original user query

    # ── Routing ───────────────────────────────────────────────────────────────
    route: Optional[str]              # "rag" | "finetuned" — set by adaptive router
    route_reasoning: Optional[str]    # Short explanation of the routing decision
    is_specific: Optional[bool]       # True = Hybrid RAG, False = Enricher first

    # ── Enrichment ────────────────────────────────────────────────────────────
    enriched_query: Optional[str]     # Query after autonomous enrichment
    assumptions: Optional[list[str]]  # What the enricher inferred (shown to user)
    web_search_results: Optional[str] # Raw web search context

    # ── Retrieval ─────────────────────────────────────────────────────────────
    source_documents: Optional[list]  # Retrieved docs from Hybrid RAG
    docs_relevant: Optional[bool]     # True = docs match query, False = fall back to fine-tuned

    # ── Output ────────────────────────────────────────────────────────────────
    answer: Optional[str]             # Final generated answer
    model_used: Optional[str]         # Which model produced the answer
    path_taken: Optional[list[str]]   # Audit trail of nodes visited
    error: Optional[str]              # Non-None when a node fails gracefully
