"""
Pipeline benchmark using pipeline_eval_dataset.json.
Samples N queries per type (vague/specific/general), runs them through
the RAG pipeline, and evaluates against ground truth disease labels.

Output: tests/performance/results.json
"""

import json
import time
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.agents.graph import run_diagnosis

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "../evaluation/pipeline_eval_dataset.json"
)
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")

# Samples per query type
SAMPLES_PER_TYPE = 8
SEED = 42

# Expected route per query type
EXPECTED_ROUTE = {
    "vague": "rag",       # should enrich then retrieve
    "specific": "rag",    # should retrieve directly
    "general": "finetuned",
}


def answer_contains_disease(answer: str, ground_truth: dict) -> bool:
    answer_lower = answer.lower()
    targets = ground_truth.get("diseases", []) + ground_truth.get("key_terms", [])
    return any(t.lower() in answer_lower for t in targets if t)


def run_eval(samples_per_type: int = SAMPLES_PER_TYPE):
    random.seed(SEED)

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    # Stratified sample
    by_type: dict[str, list] = {}
    for item in dataset:
        by_type.setdefault(item["query_type"], []).append(item)

    selected = []
    for qtype, items in by_type.items():
        k = min(samples_per_type, len(items))
        selected.extend(random.sample(items, k))

    print(f"Running {len(selected)} queries "
          f"({samples_per_type} per type × {len(by_type)} types)…\n")

    results = []
    for i, item in enumerate(selected, 1):
        qtype = item["query_type"]
        query = item["query"]
        gt = item.get("ground_truth", {})
        expected_route = EXPECTED_ROUTE.get(qtype)

        print(f"[{i}/{len(selected)}] [{qtype}] {query[:70]}…")
        start = time.time()
        try:
            state = run_diagnosis(query, user_context=None)
            elapsed = round(time.time() - start, 2)
            answer = state.get("answer") or ""
            route = state.get("route")
            route_correct = (expected_route is None) or (route == expected_route)
            disease_hit = answer_contains_disease(answer, gt)

            results.append({
                "id": i,
                "sample_id": item["sample_id"],
                "query_type": qtype,
                "query": query,
                "primary_disease": gt.get("primary_disease", ""),
                "expected_route": expected_route,
                "route": route,
                "route_correct": route_correct,
                "path_taken": state.get("path_taken", []),
                "is_specific": state.get("is_specific"),
                "docs_relevant": state.get("docs_relevant"),
                "model_used": state.get("model_used"),
                "disease_hit": disease_hit,
                "answer_snippet": answer[:250],
                "latency_s": elapsed,
                "error": state.get("error"),
            })
            print(f"  ✓ route={route} (expected={expected_route}, correct={route_correct}) "
                  f"disease_hit={disease_hit} [{elapsed}s]")
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            results.append({
                "id": i,
                "sample_id": item["sample_id"],
                "query_type": qtype,
                "query": query,
                "primary_disease": gt.get("primary_disease", ""),
                "expected_route": expected_route,
                "route": None,
                "route_correct": False,
                "path_taken": [],
                "is_specific": None,
                "docs_relevant": None,
                "model_used": None,
                "disease_hit": False,
                "answer_snippet": "",
                "latency_s": elapsed,
                "error": str(e),
            })
            print(f"  ✗ error: {e}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results → {RESULTS_PATH}")

    # Summary
    n = len(results)
    route_acc = sum(1 for r in results if r["route_correct"]) / n
    disease_acc = sum(1 for r in results if r["disease_hit"]) / n
    avg_lat = sum(r["latency_s"] for r in results) / n
    errors = sum(1 for r in results if r["error"])
    print(f"\nSummary: route_accuracy={route_acc:.0%} disease_hit_rate={disease_acc:.0%} "
          f"avg_latency={avg_lat:.1f}s errors={errors}/{n}")
    return results


if __name__ == "__main__":
    run_eval()
