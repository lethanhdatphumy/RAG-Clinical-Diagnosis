from typing import TypedDict, Optional


class AgentState(TypedDict):
    """Shared state that flows through every node of the LangGraph."""

    query: str                        # Original user query
    route: Optional[str]              # "rag" | "finetuned" — set by the router node
    route_reasoning: Optional[str]    # Short explanation of the routing decision
    answer: Optional[str]             # Final generated answer
    source_documents: Optional[list]  # Retrieved docs (RAG path only)
    model_used: Optional[str]         # Name of the model that produced the answer
    error: Optional[str]              # Non-None when a node fails gracefully
