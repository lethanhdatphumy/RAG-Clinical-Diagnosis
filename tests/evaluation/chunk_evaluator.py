"""
Chunked-Context Retrieval Evaluator.

Loads the synthetic evaluation dataset, queries Qdrant for each sample, and
computes Precision@K, Recall@K, MRR, and Hit@K.

Usage:
    python -m tests.evaluation.chunk_evaluator            # K=5 (default)
    python -m tests.evaluation.chunk_evaluator --k 10
    python -m tests.evaluation.chunk_evaluator --k 5 --save
"""

import argparse
import json
import logging
import re
from pathlib import Path
from datetime import datetime

from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient

from config.settings import Config
from tests.evaluation.synthetic_dataset import SyntheticDatasetGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("tests/results")

# ── Query cleaning ────────────────────────────────────────────────────────────

_PREFIX_RE = re.compile(
    r"^(\*?Final (?:selection|choice|string|polish|answer|check|draft)[:\*]*\s*"
    r"|Query:\s*|Result:\s*|Only query:\s*|Let's go with:\s*"
    r"|-\s*Only query:\s*(?:Yes\.)?|This is the most realistic\.)",
    re.IGNORECASE,
)


def _clean_query(raw: str) -> str:
    """Extract the actual clinical query from a verbose Gemini response."""
    # Take last non-empty line
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    last = lines[-1] if lines else raw.strip()

    # Strip known prefixes (e.g. "Final selection: ...", "Query: ...")
    last = _PREFIX_RE.sub("", last).strip().strip('"').strip("*").strip()

    # Remove enclosing quotes if present
    if last.startswith('"') and last.endswith('"'):
        last = last[1:-1].strip()

    # De-duplicate: if text is repeated back-to-back, keep the first half
    n = len(last)
    if n > 0 and n % 2 == 0:
        mid = n // 2
        if last[:mid] == last[mid:]:
            return last[:mid].strip()

    # Fallback: check if second half of text exactly matches a suffix of first half
    # (handles slight length mismatches due to trailing punctuation)
    for split in range(n // 3, 2 * n // 3):
        if last[split:] == last[:n - split]:
            return last[:split].strip()

    return last


# ── Qdrant retriever ──────────────────────────────────────────────────────────

class QdrantRetriever:
    """Thin wrapper around the raw Qdrant client so we can read flat payloads."""

    def __init__(self):
        self.embedder = HuggingFaceEmbeddings(
            model_name=Config.TEXT_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        self.client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)

    def retrieve(self, query: str, k: int) -> list[str]:
        """Embed query, search Qdrant, return list of case_ids."""
        vector = self.embedder.embed_query(query)
        results = self.client.search(
            collection_name=Config.QDRANT_COLLECTION_NAME,
            query_vector=vector,
            limit=k,
            with_payload=True,
        )
        return [r.payload.get("case_id", "") for r in results]


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved: list[str], relevant: set[str]) -> float:
    return 1.0 if any(r in relevant for r in retrieved) else 0.0


# ── Evaluator ─────────────────────────────────────────────────────────────────

def evaluate(k: int = 5) -> dict:
    samples = SyntheticDatasetGenerator.load()
    logger.info("Loaded %d samples. Building Qdrant retriever…", len(samples))

    retriever = QdrantRetriever()
    logger.info("Qdrant connected. Running evaluation at K=%d…", k)

    per_sample = []
    precision_scores, recall_scores, rr_scores, hit_scores = [], [], [], []

    for i, sample in enumerate(samples):
        raw_query = sample["query"]
        query = _clean_query(raw_query)
        relevant = set(sample["relevant_case_ids"])

        retrieved = retriever.retrieve(query, k)

        p = precision_at_k(retrieved, relevant)
        r = recall_at_k(retrieved, relevant)
        rr = reciprocal_rank(retrieved, relevant)
        h = hit_at_k(retrieved, relevant)

        precision_scores.append(p)
        recall_scores.append(r)
        rr_scores.append(rr)
        hit_scores.append(h)

        per_sample.append({
            "sample_id": sample["sample_id"],
            "source_case_id": sample["source_case_id"],
            "is_clinical": sample["is_clinical"],
            "cleaned_query": query,
            "retrieved_case_ids": retrieved,
            "relevant_case_ids": list(relevant),
            "precision_at_k": round(p, 4),
            "recall_at_k": round(r, 4),
            "reciprocal_rank": round(rr, 4),
            "hit_at_k": h,
        })

        logger.info(
            "[%d/%d] %s | P@%d=%.3f R@%d=%.3f RR=%.3f Hit=%d",
            i + 1, len(samples), sample["sample_id"],
            k, p, k, r, rr, int(h),
        )

    n = len(samples)
    aggregate = {
        "k": k,
        "n_samples": n,
        "mean_precision_at_k": round(sum(precision_scores) / n, 4),
        "mean_recall_at_k": round(sum(recall_scores) / n, 4),
        "mrr": round(sum(rr_scores) / n, 4),
        "hit_rate_at_k": round(sum(hit_scores) / n, 4),
    }

    # Breakdown: clinical vs metadata
    clinical = [s for s in per_sample if s["is_clinical"]]
    metadata = [s for s in per_sample if not s["is_clinical"]]

    def _agg(subset):
        if not subset:
            return {}
        nc = len(subset)
        return {
            "n": nc,
            "mean_precision_at_k": round(sum(s["precision_at_k"] for s in subset) / nc, 4),
            "mean_recall_at_k": round(sum(s["recall_at_k"] for s in subset) / nc, 4),
            "mrr": round(sum(s["reciprocal_rank"] for s in subset) / nc, 4),
            "hit_rate_at_k": round(sum(s["hit_at_k"] for s in subset) / nc, 4),
        }

    return {
        "aggregate": aggregate,
        "clinical_cases": _agg(clinical),
        "metadata_pages": _agg(metadata),
        "per_sample": per_sample,
    }


def _print_report(results: dict) -> None:
    agg = results["aggregate"]
    k = agg["k"]
    print(f"\n{'='*60}")
    print(f"  Chunked-Context Retrieval Evaluation — K={k}")
    print(f"{'='*60}")
    print(f"  Samples evaluated : {agg['n_samples']}")
    print(f"  Precision@{k:<3}     : {agg['mean_precision_at_k']:.4f}")
    print(f"  Recall@{k:<5}       : {agg['mean_recall_at_k']:.4f}")
    print(f"  MRR               : {agg['mrr']:.4f}")
    print(f"  Hit Rate@{k:<3}     : {agg['hit_rate_at_k']:.4f}")

    for label, key in [("Clinical cases", "clinical_cases"), ("Metadata pages", "metadata_pages")]:
        sub = results.get(key, {})
        if sub:
            print(f"\n  ── {label} (n={sub['n']}) ──")
            print(f"     Precision@{k}: {sub['mean_precision_at_k']:.4f}")
            print(f"     Recall@{k}  : {sub['mean_recall_at_k']:.4f}")
            print(f"     MRR        : {sub['mrr']:.4f}")
            print(f"     Hit Rate@{k}: {sub['hit_rate_at_k']:.4f}")

    print(f"\n{'='*60}\n")


def _save_results(results: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"chunk_eval_k{results['aggregate']['k']}_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", path)
    return path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate chunked-context retrieval")
    parser.add_argument("--k", type=int, default=5, help="Top-K to retrieve (default: 5)")
    parser.add_argument("--save", action="store_true", help="Save results JSON to tests/results/")
    args = parser.parse_args()

    results = evaluate(k=args.k)
    _print_report(results)

    if args.save:
        path = _save_results(results)
        print(f"Saved → {path}")


if __name__ == "__main__":
    main()
