import logging
from src.generation.rag_generator import ClinicalRAG
from src.agents.state import AgentState
from config.settings import Config

logger = logging.getLogger(__name__)

_rag_instance: ClinicalRAG | None = None


def _get_rag() -> ClinicalRAG:
    """Lazy-load the RAG system (heavy; load once per process)."""
    global _rag_instance
    if _rag_instance is None:
        logger.info("Initialising ClinicalRAG...")
        _rag_instance = ClinicalRAG()
    return _rag_instance


def rag_node(state: AgentState) -> AgentState:
    """
    Runs the FAISS-backed RAG pipeline and writes the answer into state.
    """
    query = state["query"]
    logger.info("RAG node — query: %s", query[:80])

    try:
        rag = _get_rag()
        result = rag.query(query)

        answer = result.get("result", "")
        source_docs = result.get("source_documents", [])

        return {
            **state,
            "answer": answer,
            "source_documents": source_docs,
            "model_used": f"RAG + {Config.GEMINI_MODEL}",
        }
    except Exception as exc:
        logger.error("RAG node error: %s", exc)
        return {
            **state,
            "answer": None,
            "error": f"RAG pipeline failed: {exc}",
            "model_used": f"RAG + {Config.GEMINI_MODEL}",
        }
