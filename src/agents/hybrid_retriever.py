"""
Hybrid Retriever Node.

Combines:
  - Dense search  : Qdrant (semantic, all-MiniLM-L6-v2 embeddings)
  - Sparse search : BM25 (keyword, built from filtered JSON files)
  - Fusion        : Reciprocal Rank Fusion (RRF)

The BM25 index is built once at startup from Qdrant payloads and cached.
"""

import logging
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from config.settings import Config
from src.agents.state import AgentState

logger = logging.getLogger(__name__)

# ── BM25 index (built once, cached in module) ─────────────────────────────────

_bm25: BM25Okapi | None = None
_bm25_docs: list[dict] | None = None   # parallel list of payloads


def _build_bm25() -> tuple[BM25Okapi, list[dict]]:
    """Fetch all payloads from Qdrant and build a BM25 index."""
    global _bm25, _bm25_docs
    if _bm25 is not None:
        return _bm25, _bm25_docs

    logger.info("Building BM25 index from Qdrant payloads…")
    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)

    # Scroll through all points in the collection
    docs = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            docs.append(point.payload)
        if offset is None:
            break

    # Tokenise: combine page_content + diseases + symptoms for better keyword coverage
    tokenised = []
    for doc in docs:
        text = " ".join(filter(None, [
            doc.get("page_content", ""),
            " ".join(doc.get("diseases", [])),
            " ".join(doc.get("symptoms", [])),
        ]))
        tokenised.append(text.lower().split())

    _bm25 = BM25Okapi(tokenised)
    _bm25_docs = docs
    logger.info("BM25 index built — %d documents.", len(docs))
    return _bm25, _bm25_docs


# ── RRF fusion ────────────────────────────────────────────────────────────────

def _rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """
    Reciprocal Rank Fusion.
    rankings: list of ranked lists of case_ids (best first).
    Returns a single merged ranked list.
    """
    scores: dict[str, float] = {}
    for ranked in rankings:
        for rank, case_id in enumerate(ranked, start=1):
            scores[case_id] = scores.get(case_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


# ── Node ──────────────────────────────────────────────────────────────────────

def hybrid_retriever_node(state: AgentState) -> AgentState:
    """
    Retrieves top-K documents using Hybrid RAG (BM25 + Qdrant + RRF).
    Uses enriched_query if available, otherwise falls back to original query.
    """
    query = state.get("enriched_query") or state["query"]
    path = list(state.get("path_taken") or []) + ["hybrid_retriever"]
    k = Config.TOP_K_RETRIEVAL
    logger.info("Hybrid retriever — query: %s", query[:80])

    try:
        # ── Dense: Qdrant ──────────────────────────────────────────────────
        embedder = HuggingFaceEmbeddings(
            model_name=Config.TEXT_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
        vector = embedder.embed_query(query)

        dense_results = client.search(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            query_vector=vector,
            limit=k * 2,
            with_payload=True,
        )
        dense_ids = [r.payload.get("case_id", "") for r in dense_results]
        dense_payloads = {r.payload.get("case_id", ""): r.payload for r in dense_results}

        # ── Sparse: BM25 ───────────────────────────────────────────────────
        bm25, bm25_docs = _build_bm25()
        tokens = query.lower().split()
        bm25_scores = bm25.get_scores(tokens)
        top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k * 2]
        sparse_ids = [bm25_docs[i].get("case_id", "") for i in top_bm25_idx]
        sparse_payloads = {bm25_docs[i].get("case_id", ""): bm25_docs[i] for i in top_bm25_idx}

        # ── RRF merge ──────────────────────────────────────────────────────
        merged_ids = _rrf([dense_ids, sparse_ids])[:k]
        all_payloads = {**sparse_payloads, **dense_payloads}
        docs = [all_payloads[cid] for cid in merged_ids if cid in all_payloads]

        logger.info("Hybrid retriever — %d docs retrieved (dense=%d, sparse=%d, merged=%d)",
                    len(docs), len(dense_ids), len(sparse_ids), len(merged_ids))

        return {**state, "source_documents": docs, "path_taken": path}

    except Exception as exc:
        logger.error("Hybrid retriever error: %s", exc)
        return {**state, "source_documents": [], "error": str(exc), "path_taken": path}
